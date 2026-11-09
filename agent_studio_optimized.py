import ollama
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
import json
import re
import io
from contextlib import redirect_stdout
from collections import deque
import threading
import time
from queue import Queue



# 日志配置（记录流程，无思考模式）
# 使用绝对路径确保日志文件在正确的位置
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_studio.log")

# 手动配置日志，而不是使用basicConfig
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 清除现有的处理器
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 创建文件处理器
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# 添加处理器到根日志器
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# 获取智能体工作室日志器
logger = logging.getLogger("智能体工作室")

# 模型常量定义（与设计方案一致，便于维护）
MODEL_MANAGER = "qwen3:1.7b"  # 管理单元模型
MODEL_INTENT = "qwen3:0.6b"  # 意图识别单元模型（使用轻量模型提高速度）
MODEL_EXEC_ADVANCED = "qwen3:4b-instruct"  # 高级执行单元模型
MODEL_EXEC_CORE = "qwen2.5:3b-instruct"  # 核心执行单元模型
MODEL_EXEC_ADVANCED_CODER = "qwen2.5-coder:1.5b"  # 高级编程单元模型
MODEL_EXEC_QUICK_CODER = "qwen2.5-coder:1.5b-instruct-q4_0"  # 快速编程单元模型
MODEL_EXEC_LIGHT = "qwen2.5:1.5b"  # 轻量执行单元模型
MODEL_EXEC_CHAT = "qwen3:0.6b"  # 对话执行单元模型
MODEL_EXEC_SIMPLE = "qwen2.5:0.5b"  # 极简执行单元模型
MODEL_EXEC_MINIMAL_CODER = "qwen2.5-coder:0.5b"  # 极简编程单元模型

# 角色复杂度枚举（管理单元用于自动匹配模型）
class RoleComplexity:
    ADVANCED = "advanced"  # 深度分析/复杂创作/逻辑推理
    ADVANCED_CODER = "advanced_coder"  # 高质量代码生成/复杂编程任务/算法设计
    CORE = "core"  # 分析/创作/逻辑处理
    QUICK_CODER = "quick_coder"  # 快速代码生成/简单编程任务/脚本编写
    LIGHT = "light"  # 整理/加工/格式转换
    CHAT = "chat"  # 对话沟通/简单问答/咨询
    SIMPLE = "simple"  # 检索/校对/基础执行
    MINIMAL_CODER = "minimal_coder"  # 简单代码生成/脚本编写/代码调试

# 数据类：子任务+角色信息（结构化，便于传递）
@dataclass
class TaskRole:
    task_id: str               # 子任务ID（唯一）
    role_name: str             # 角色名称
    role_duty: str             # 角色职责
    execute_way: str           # 执行方式
    output_format: str         # 输出格式
    sub_task: str              # 子任务内容
    complexity: str            # 角色复杂度
    dependencies: List[str]     # 依赖的任务ID列表，空列表表示无依赖

# 数据类：执行单元结果
@dataclass
class ExecResult:
    task_id: str       # 对应子任务ID
    role_name: str     # 执行角色
    model_used: str    # 使用的Ollama模型
    result: str        # 子任务执行结果
    success: bool      # 是否执行成功

class ManagementUnit:
    """管理控制单元（qwen3:1.7b）- 无思考模式，固定流程执行"""
    def __init__(self, model_name: str = MODEL_MANAGER):
        self.model = model_name
        self.logger = logging.getLogger("管理控制单元")

    def task_split_and_role_define(self, user_task: str) -> List[TaskRole]:
        """
        核心方法：任务拆解+角色定义（无思考，按固定prompt让模型输出结构化结果）
        :param user_task: 用户原始任务
        :return: 结构化的子任务+角色列表
        """
        # 固定prompt（无思考，强制模型按指定格式输出，保证后续解析）
        prompt = """
        你是智能体工作室的管理控制单元，仅负责**任务拆解**和**角色定义**，不参与任何子任务执行。
        请严格按以下要求处理用户任务：""" + user_task + """
        1. 任务拆解：将用户任务拆分为多个子任务，子任务数量按需定，无上限。对于可以合并为一个任务完成的工作，不要分解成多个任务。
        2. 依赖关系：分析子任务间的依赖关系，明确哪些任务可以并行执行，哪些任务有先后次序。如果某些任务需要其他任务的结果，必须标注依赖关系。
        3. 角色定义：为每个子任务定义唯一角色，包含【角色名称、核心职责、执行方式、输出格式】，描述需具体、可落地；
        4. 复杂度标注：为每个角色标注复杂度（advanced/advanced_coder/core/quick_coder/light/chat/simple/minimal_coder），规则：
            - advanced：需要深度分析、复杂创作、逻辑推理的高级角色；
            - advanced_coder：需要高质量代码生成、复杂编程任务、算法设计的角色；
            - core：需要分析、创作、逻辑推理、深度处理的角色；
            - quick_coder：需要快速代码生成、简单编程任务、脚本编写的角色；
            - light：仅需要内容整理、格式转换、简单加工的角色；
            - chat：需要对话沟通、简单问答、咨询的角色；
            - simple：仅需要信息检索、结果校对、基础执行、纯输出的角色；
            - minimal_coder：需要简单代码生成、脚本编写、代码调试的角色；
         5. 特殊处理：对于数学计算任务，必须拆解出一个专门的"计算任务单元"，负责生成计算代码脚本，使用advanced_coder或quick_coder复杂度；
         6. 输出格式：严格按以下JSON格式输出，无额外内容，无解释，键名不可修改：
         [
             {
                 "task_id": "t1",
                 "role_name": "XXX角色",
                 "role_duty": "XXX",
                 "execute_way": "XXX",
                 "output_format": "XXX",
                 "sub_task": "XXX子任务内容",
                 "complexity": "advanced/advanced_coder/core/quick_coder/light/chat/simple/minimal_coder",
                 "dependencies": ["t2", "t3"]
             }
         ]
        注意：
        - dependencies字段为数组，表示当前任务依赖的其他任务ID。如果无依赖，填写空数组[]。
        - 对于可以并行执行的任务，确保它们之间没有依赖关系。
        - 对于有先后次序的任务，确保正确标注依赖关系。
        """
        # 调用Ollama模型获取拆解结果
        try:
            self.logger.info("开始拆解任务：" + user_task)
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是智能体工作室的管理控制单元，仅负责任务拆解和角色定义，不参与任何子任务执行。请严格按指定格式输出JSON结果。"},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "num_ctx": 4096}
            )
            
            # 提取JSON响应
            content = response['message']['content']
            # 尝试从内容中提取JSON
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                tasks_data = json.loads(json_str)
                
                task_roles = []
                for item in tasks_data:
                    # 确保字符串字段始终是字符串类型
                    def ensure_string(value):
                        if isinstance(value, list):
                            return " ".join(str(item) for item in value)
                        elif value is None:
                            return ""
                        else:
                            return str(value)
                    
                    task_role = TaskRole(
                        task_id=ensure_string(item["task_id"]),
                        role_name=ensure_string(item["role_name"]),
                        role_duty=ensure_string(item["role_duty"]),
                        execute_way=ensure_string(item["execute_way"]),
                        output_format=ensure_string(item["output_format"]),
                        sub_task=ensure_string(item["sub_task"]),
                        complexity=ensure_string(item.get("complexity", "simple")),
                        dependencies=item.get("dependencies", [])
                    )
                    task_roles.append(task_role)
                
                self.logger.info("任务拆解完成，共生成" + str(len(task_roles)) + "个子任务/角色")
                return task_roles
            else:
                self.logger.error("模型输出格式错误，未找到JSON数组")
                raise Exception("模型输出格式错误，未找到JSON数组")
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON解析失败：" + str(e), exc_info=True)
            raise Exception("任务拆解结果解析失败：" + str(e))
        except Exception as e:
            self.logger.error("任务拆解失败：" + str(e), exc_info=True)
            raise Exception("管理单元任务拆解失败：" + str(e))

    def complexity_to_model(self, complexity: str) -> str:
        """自动匹配：角色复杂度 → 执行单元模型（无思考，固定映射）"""
        model_map = {
            "advanced": MODEL_EXEC_ADVANCED,
            "advanced_coder": MODEL_EXEC_ADVANCED_CODER,
            "core": MODEL_EXEC_CORE,
            "quick_coder": MODEL_EXEC_QUICK_CODER,
            "light": MODEL_EXEC_LIGHT,
            "chat": MODEL_EXEC_CHAT,
            "simple": MODEL_EXEC_SIMPLE,
            "minimal_coder": MODEL_EXEC_MINIMAL_CODER
        }
        return model_map.get(complexity, MODEL_EXEC_SIMPLE)

    def summarize_results(self, exec_results: List[ExecResult], user_task: str) -> str:
        """
        核心方法：结果汇总分析（无思考，按固定prompt让模型汇总）
        :param exec_results: 所有执行单元的结果
        :param user_task: 用户原始任务
        :return: 汇总后的最终答复
        """
        # 整理执行结果为结构化字符串
        results_str_parts = []
        for res in exec_results:
            if res.success:
                part = "【子任务ID：" + res.task_id + "】\n角色：" + res.role_name + "\n使用模型：" + res.model_used + "\n执行结果：" + res.result
                results_str_parts.append(part)
        results_str = "\n".join(results_str_parts)
        
        # 固定汇总prompt（无思考，强制按用户易读格式汇总）
        prompt = """
        你是智能体工作室的管理控制单元，负责将所有执行单元的结果汇总分析后答复用户。
        原始用户任务：""" + user_task + """
        各执行单元执行结果：
        """ + results_str + """
        汇总要求：
        1. 按子任务重要性排序，整合所有有效结果，无重复内容；
        2. 保持结果的完整性和准确性，严格遵循各执行单元的输出内容；
        3. 格式清晰，分点/分模块呈现，便于用户阅读；
        4. 无需额外解释，直接给出针对用户原始任务的最终答复。
        """
        # 调用Ollama模型生成汇总结果
        try:
            self.logger.info("开始汇总分析执行结果")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是智能体工作室的管理控制单元，负责将所有执行单元的结果汇总分析后答复用户。请直接给出最终答复，无需额外解释。"},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "num_ctx": 4096}
            )
            
            self.logger.info("结果汇总分析完成，生成最终答复")
            return response['message']['content']
        except Exception as e:
            self.logger.error("结果汇总失败：" + str(e), exc_info=True)
            raise Exception("管理单元结果汇总失败：" + str(e))

class ExecUnit:
    """执行单元（动态创建，无数量限制）- 无思考模式，按要求执行"""
    def __init__(self, task_role: TaskRole, model: str, progress_callback=None):
        self.task_role = task_role  # 绑定子任务+角色
        self.base_model = model     # 基础Ollama模型
        self.task_id = task_role.task_id
        self.role_name = task_role.role_name
        self.model_name = "executor_" + self.task_id  # 创建的模型名称
        self.logger = logging.getLogger("执行单元-" + self.task_id + "-" + self.role_name)
        self.created_model = False
        self.progress_callback = progress_callback  # 进度回调函数
        
        # 创建基于角色的ollama模型
        self._create_model()
        
        # 汇报任务开始
        if self.progress_callback:
            self.progress_callback(self.task_id, self.role_name, "pending", "任务等待执行")

    def _create_model(self):
        """创建基于角色的ollama模型"""
        try:
            self.logger.info("开始创建模型：" + self.model_name + "，基于：" + self.base_model)
            # 构建系统提示
            system_prompt = "你的唯一身份是：" + self.role_name + "\n你的核心职责是：" + self.task_role.role_duty + "\n你的执行方式是：" + self.task_role.execute_way
            # 使用create方法创建模型
            ollama.create(
                model=self.model_name,
                from_=self.base_model,
                system=system_prompt,
                parameters={"temperature": 0.0, "num_ctx": 4096}
            )
            self.created_model = True
            self.logger.info("成功创建模型：" + self.model_name)
        except Exception as e:
            # 记录详细的创建参数
            system_prompt = "你的唯一身份是：" + self.role_name + "\n你的核心职责是：" + self.task_role.role_duty + "\n你的执行方式是：" + self.task_role.execute_way
            self.logger.error(f"创建模型时出错：模型名称='{self.model_name}'，基础模型='{self.base_model}'，系统提示='{system_prompt[:100]}...'，错误信息='{str(e)}'，将使用基础模型")
            self.created_model = False

    def execute_task(self) -> ExecResult:
        """核心方法：执行子任务（无思考，严格按角色和要求执行）"""
        # 汇报任务开始执行
        if self.progress_callback:
            self.progress_callback(self.task_id, self.role_name, "running", "任务正在执行中")
        
        # 确定使用的模型
        model_to_use = self.model_name if self.created_model else self.base_model
        
        # 构建任务输入
        # 确保字符串拼接安全
        def safe_str(value):
            if isinstance(value, list):
                return " ".join(str(item) for item in value)
            elif value is None:
                return ""
            else:
                return str(value)
        
        task_input = "请严格按你的身份和要求，完成以下子任务：" + safe_str(self.task_role.sub_task) + "\n输出要求：严格按指定格式输出，无额外内容，无解释，输出格式为：" + safe_str(self.task_role.output_format)
        
        try:
            self.logger.info("开始执行子任务，使用模型：" + model_to_use)
            self.logger.info("任务输入：" + task_input)
            
            # 检查是否是数学计算任务
            if "计算" in self.role_name or "数学" in self.role_name:
                result = self._execute_math_task(model_to_use, task_input)
            else:
                response = ollama.chat(
                    model=model_to_use,
                    messages=[
                        {"role": "user", "content": task_input}
                    ],
                    options={"temperature": 0.0, "num_ctx": 4096}
                )
                
                task_output = response['message']['content']
                self.logger.info("子任务执行完成，已返回结果")
                self.logger.info("任务输出：" + task_output)
                
                result = ExecResult(
                    task_id=self.task_id,
                    role_name=self.role_name,
                    model_used=model_to_use,
                    result=task_output,
                    success=True
                )
            
            # 汇报任务完成
            if self.progress_callback:
                if result.success:
                    self.progress_callback(self.task_id, self.role_name, "completed", "任务执行完成")
                else:
                    self.progress_callback(self.task_id, self.role_name, "failed", "任务执行失败")
            
            return result
        except Exception as e:
            self.logger.error("子任务执行失败：" + str(e), exc_info=True)
            error_output = "执行失败：" + str(e)
            self.logger.info("任务输出：" + error_output)
            
            # 汇报任务失败
            if self.progress_callback:
                self.progress_callback(self.task_id, self.role_name, "failed", f"任务执行失败：{str(e)}")
            
            return ExecResult(
                task_id=self.task_id,
                role_name=self.role_name,
                model_used=model_to_use,
                result=error_output,
                success=False
            )
    
    def _execute_math_task(self, model_to_use: str, task_input: str) -> ExecResult:
        """执行数学计算任务，生成代码并执行"""
        try:
            # 生成计算代码脚本
            self.logger.info("生成数学计算代码脚本")
            code_gen_input = "请为以下数学计算任务生成Python代码脚本：" + self.task_role.sub_task + "\n要求：\n1. 代码必须能正确计算结果\n2. 代码必须包含打印结果的语句\n3. 代码必须是完整的可执行脚本\n4. 只输出代码，不输出其他内容"
            self.logger.info("代码生成输入：" + code_gen_input)
            
            response = ollama.chat(
                model=model_to_use,
                messages=[
                    {"role": "user", "content": code_gen_input}
                ],
                options={"temperature": 0.0, "num_ctx": 4096}
            )
            
            code_script = response['message']['content']
            self.logger.info("生成的代码脚本：" + code_script)
            
            # 执行代码脚本
            self.logger.info("执行数学计算代码脚本")
            exec_result = self._run_python_code(code_script)
            
            self.logger.info("数学计算任务完成，结果：" + exec_result)
            self.logger.info("任务输出：" + exec_result)
            
            return ExecResult(
                task_id=self.task_id,
                role_name=self.role_name,
                model_used=model_to_use,
                result=exec_result,
                success=True
            )
        except Exception as e:
            self.logger.error("数学计算任务执行失败：" + str(e), exc_info=True)
            error_output = "执行失败：" + str(e)
            self.logger.info("任务输出：" + error_output)
            return ExecResult(
                task_id=self.task_id,
                role_name=self.role_name,
                model_used=model_to_use,
                result=error_output,
                success=False
            )
    
    def _run_python_code(self, code: str) -> str:
        """执行Python代码并返回结果"""
        try:
            # 提取代码部分（如果有代码块标记）
            code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            
            # 创建临时执行环境
            local_vars = {}
            # 重定向标准输出以捕获打印结果
            
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, globals(), local_vars)
            
            output = f.getvalue().strip()
            
            # 如果没有输出，尝试获取最后一个表达式的值
            if not output:
                # 尝试执行代码并获取结果
                lines = code.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    if not last_line.strip().startswith('#') and not last_line.strip().endswith(':'):
                        try:
                            result = eval(last_line, globals(), local_vars)
                            output = str(result)
                        except:
                            pass
            
            return output if output else "执行成功，但未获取到结果"
        except Exception as e:
            return "代码执行错误：" + str(e)
    
    def cleanup(self):
        """清理资源，删除创建的模型"""
        if self.created_model:
            try:
                self.logger.info("开始删除模型：" + self.model_name)
                ollama.delete(self.model_name)
                self.logger.info("成功删除模型：" + self.model_name)
            except Exception as e:
                self.logger.error("删除模型时出错：" + str(e))

class AgentStudio:
    """智能体工作室（主程序）- 整合管理单元+执行单元，调度整个流程"""
    def __init__(self):
        self.logger = logging.getLogger("智能体工作室")
        self.task_monitor = None  # 任务监控单元引用
        self.task_progress_callback = None  # 任务进度回调函数
        
        # 运行时模型名称
        self.management_model_name = "management-unit"  # 管理单元专用模型
        self.intent_model_name = "intent-recognition-unit"  # 意图识别单元专用模型
        
        # 创建运行时模型
        self._create_runtime_models()
        
        # 使用运行时模型初始化管理单元
        self.management_unit = ManagementUnit(self.management_model_name)
    
    def _create_runtime_models(self):
        """创建运行时专用模型"""
        try:
            # 创建管理单元专用模型
            self.logger.info(f"正在创建管理单元专用模型：{self.management_model_name}")
            ollama.create(
                model=self.management_model_name,
                from_=MODEL_MANAGER,
                system="你是智能体工作室的管理控制单元，负责任务拆解和角色定义。你的职责是：\n1. 接收用户的任务要求，分析任务的复杂度和依赖关系\n2. 将任务拆解为多个子任务，每个子任务对应一个执行单元\n3. 为每个执行单元定义角色名称、职责、执行方式和输出格式\n4. 确定任务之间的依赖关系，确保执行顺序正确\n5. 严格按照JSON格式输出任务规划\n\n请始终保持管理单元的身份，不要自称其他身份。"
            )
            self.logger.info(f"管理单元专用模型创建成功：{self.management_model_name}")
            
            # 创建意图识别单元专用模型
            self.logger.info(f"正在创建意图识别单元专用模型：{self.intent_model_name}")
            ollama.create(
                model=self.intent_model_name,
                from_=MODEL_INTENT,
                system="你是智能体工作室的意图识别单元，负责判断用户输入的意图。你的职责是：\n1. 准确识别用户是要执行任务、简单对话还是查询任务监控\n2. 只输出\"TASK\"、\"CHAT\"或\"MONITOR\"中的一个词\n3. 不要输出任何其他内容或解释\n\n判断标准：\n- TASK：用户要求执行具体任务（如计算、创作、分析、编程、翻译等）\n- CHAT：用户进行简单对话（如问候、闲聊、咨询、感谢等）\n- MONITOR：用户查询任务监控信息（如询问任务进度、状态、结果等）\n\n请只输出一个词，不要输出其他内容。"
            )
            self.logger.info(f"意图识别单元专用模型创建成功：{self.intent_model_name}")
            
        except Exception as e:
            self.logger.error(f"创建运行时模型失败：{str(e)}")
            # 如果创建失败，使用原模型作为后备
            self.management_model_name = MODEL_MANAGER
            self.intent_model_name = MODEL_INTENT
    
    def _cleanup_runtime_models(self):
        """清理运行时专用模型"""
        try:
            # 删除管理单元专用模型
            if self.management_model_name != MODEL_MANAGER:
                self.logger.info(f"正在删除管理单元专用模型：{self.management_model_name}")
                ollama.delete(self.management_model_name)
                self.logger.info(f"管理单元专用模型删除成功：{self.management_model_name}")
            
            # 删除意图识别单元专用模型
            if self.intent_model_name != MODEL_INTENT:
                self.logger.info(f"正在删除意图识别单元专用模型：{self.intent_model_name}")
                ollama.delete(self.intent_model_name)
                self.logger.info(f"意图识别单元专用模型删除成功：{self.intent_model_name}")
        except Exception as e:
            self.logger.error(f"删除运行时模型失败：{str(e)}")
    
    def set_task_monitor(self, task_monitor):
        """设置任务监控单元"""
        self.task_monitor = task_monitor
    
    def set_task_progress_callback(self, callback):
        """设置任务进度回调函数"""
        self.task_progress_callback = callback

    def _identify_intent(self, user_input: str) -> str:
        """
        识别用户输入意图：判断是要执行任务、简单对话沟通还是查询任务监控（使用模型语义识别）
        :param user_input: 用户输入
        :return: "TASK"表示执行任务，"CHAT"表示简单对话，"MONITOR"表示任务监控
        """
        try:
            self.logger.info("开始识别用户意图")
            # 使用意图识别单元专用模型进行意图识别
            response = ollama.chat(
                model=self.intent_model_name,  # 使用意图识别单元专用模型
                messages=[
                    {"role": "user", "content": user_input}
                ],
                options={"temperature": 0.0, "num_ctx": 2048}  # 减少上下文长度，提高速度
            )
            
            intent = response['message']['content'].strip()
            self.logger.info(f"用户意图识别结果：{intent}")
            
            # 确保返回有效的意图类型
            if intent in ["TASK", "CHAT", "MONITOR"]:
                return intent
            else:
                self.logger.warning(f"意图识别结果无效：{intent}，默认按对话处理")
                return "CHAT"
        except Exception as e:
            self.logger.error(f"意图识别失败：{str(e)}，默认按对话处理")
            return "CHAT"
    
    def _handle_chat(self, user_input: str) -> str:
        """
        处理简单对话沟通（流输出形式）
        :param user_input: 用户输入
        :return: 对话答复
        """
        prompt = f"""
        你是智能体工作室的对话助手，负责回答用户的简单问题和闲聊。
        用户输入：{user_input}
        请直接回答用户的问题，保持友好、自然的语气。
        """
        
        try:
            self.logger.info("开始处理简单对话")
            
            # 使用对话执行单元模型处理简单对话，开启流输出
            print("\n智能体: ", end="")
            
            # 用于存储完整的响应
            full_response = ""
            
            # 开启流输出
            stream = ollama.chat(
                model=MODEL_EXEC_CHAT,
                messages=[
                    {"role": "system", "content": "你是智能体工作室的对话助手，负责回答用户的简单问题和闲聊。请直接回答用户的问题，保持友好、自然的语气。"},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                options={"temperature": 0.7, "num_ctx": 4096}
            )
            
            # 处理流输出
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            # 确保有换行
            print()
            
            self.logger.info("简单对话处理完成")
            return full_response
        except Exception as e:
            self.logger.error(f"对话处理失败：{str(e)}")
            error_message = "抱歉，我暂时无法回答您的问题。"
            print(f"\n智能体: {error_message}")
            return error_message
    
    def run(self, user_task: str) -> str:
        """
        工作室主运行方法（无思考模式，按固定流程执行）
        :param user_task: 用户发送的任务要求
        :return: 最终答复结果
        """
        exec_units = []
        try:
            # 前置步骤：识别用户意图
            self.logger.info("开始处理用户输入，识别意图")
            if not self._identify_intent(user_task):
                # 如果是简单对话，直接答复
                return self._handle_chat(user_task)
            
            # 步骤1：管理单元 - 任务拆解+角色定义
            self.logger.info("用户意图为执行任务，进入任务拆解阶段")
            task_roles = self.management_unit.task_split_and_role_define(user_task)
            if not task_roles:
                return "错误：管理单元未拆解出任何子任务，请重新发送任务"

            # 步骤2：动态创建执行单元（一一对应，无数量限制）
            self.logger.info("开始动态创建执行单元")
            exec_units = []
            task_id_to_exec = {}
            for tr in task_roles:
                # 自动匹配模型
                model = self.management_unit.complexity_to_model(tr.complexity)
                # 创建执行单元，传入进度回调函数
                exec_unit = ExecUnit(task_role=tr, model=model, progress_callback=self.task_progress_callback)
                exec_units.append(exec_unit)
                task_id_to_exec[tr.task_id] = exec_unit

            # 步骤3：处理任务依赖关系，确定执行顺序
            self.logger.info("分析任务依赖关系，确定执行顺序")
            
            # 拓扑排序确定执行顺序
            task_order = self._topological_sort(task_roles)
            self.logger.info("任务执行顺序：" + str(task_order))

            # 步骤4：分组执行任务（并行执行无依赖的任务）
            self.logger.info("开始执行任务，处理依赖关系")
            exec_results = []
            completed_tasks = set()
            task_results = {}
            
            while completed_tasks != set(task_order):
                # 找出所有依赖已完成的任务
                ready_tasks = []
                for task_id in task_order:
                    if task_id not in completed_tasks:
                        exec_unit = task_id_to_exec[task_id]
                        task_role = exec_unit.task_role
                        # 检查所有依赖是否已完成
                        if all(dep in completed_tasks for dep in task_role.dependencies):
                            # 将依赖任务的结果传递给当前任务
                            if task_role.dependencies:
                                dependency_info = []
                                for dep_id in task_role.dependencies:
                                    if dep_id in task_results and task_results[dep_id].success:
                                        dependency_info.append(f"【依赖任务 {dep_id} 结果】：{task_results[dep_id].result}")
                                
                                if dependency_info:
                                    # 更新子任务内容，添加依赖结果
                                    dependency_text = "\n" + "\n".join(dependency_info)
                                    exec_unit.task_role.sub_task += dependency_text
                                    self.logger.info(f"已将依赖任务结果传递给任务 {task_id}")
                            
                            ready_tasks.append(task_id)
                
                if not ready_tasks:
                    break
                
                # 并行执行当前就绪的任务
                self.logger.info(f"并行执行任务：{ready_tasks}")
                with ThreadPoolExecutor(max_workers=len(ready_tasks)) as executor:
                    future_to_task = {executor.submit(task_id_to_exec[task_id].execute_task): task_id for task_id in ready_tasks}
                    for future in as_completed(future_to_task):
                        task_id = future_to_task[future]
                        try:
                            result = future.result()
                            exec_results.append(result)
                            task_results[task_id] = result
                            completed_tasks.add(task_id)
                            self.logger.info(f"任务完成：{task_id}")
                        except Exception as e:
                            self.logger.error(f"执行任务 {task_id} 时出错：{str(e)}")
                            error_result = ExecResult(
                                task_id=task_id,
                                role_name=task_id_to_exec[task_id].role_name,
                                model_used=task_id_to_exec[task_id].model_name if task_id_to_exec[task_id].created_model else task_id_to_exec[task_id].base_model,
                                result=f"执行失败：{str(e)}",
                                success=False
                            )
                            exec_results.append(error_result)
                            task_results[task_id] = error_result
                            completed_tasks.add(task_id)

            # 步骤5：管理单元 - 汇总分析结果
            self.logger.info("所有执行单元任务完成，进入结果汇总阶段")
            
            # 生成最终答复
            final_answer = self.management_unit.summarize_results(exec_results, user_task)

            return final_answer
        except Exception as e:
            self.logger.error("工作室运行失败：" + str(e), exc_info=True)
            return "智能体工作室处理任务失败：" + str(e)
        finally:
            # 清理资源，删除创建的模型
            if exec_units:
                self.logger.info("开始清理执行单元资源")
                for exec_unit in exec_units:
                    exec_unit.cleanup()
                self.logger.info("执行单元资源清理完成")
                
    def _topological_sort(self, task_roles: List[TaskRole]) -> List[str]:
        """
        对任务进行拓扑排序，确定执行顺序
        :param task_roles: 任务角色列表
        :return: 排序后的任务ID列表
        """
        # 构建任务依赖图
        task_graph = {}
        in_degree = {}
        
        # 初始化图和入度
        for tr in task_roles:
            task_id = tr.task_id
            task_graph[task_id] = tr.dependencies
            in_degree[task_id] = len(tr.dependencies)
        
        # 拓扑排序
        result = []
        queue = deque([task_id for task_id in in_degree if in_degree[task_id] == 0])
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # 遍历所有任务，更新依赖关系
            for task_id in task_graph:
                if current in task_graph[task_id]:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # 检查是否有环
        if len(result) != len(task_roles):
            self.logger.error("任务依赖关系存在环，返回原始顺序")
            return [tr.task_id for tr in task_roles]
        
        return result

@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: str
    role_name: str
    status: str  # pending, running, completed, failed
    progress: str
    timestamp: str

class TaskMonitor:
    """任务监控单元 - 实时汇报任务进展"""
    def __init__(self):
        self.logger = logging.getLogger("任务监控单元")
        self.progress_queue = Queue()
        self.monitoring = False
        self.monitor_thread = None
        self.task_progress = {}  # task_id -> TaskProgress
    
    def start_monitoring(self):
        """启动任务监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("任务监控已启动")
    
    def stop_monitoring(self):
        """停止任务监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("任务监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 检查是否有新的进度信息
                while not self.progress_queue.empty():
                    progress = self.progress_queue.get_nowait()
                    self._report_progress(progress)
                
                time.sleep(0.5)  # 每0.5秒检查一次
            except Exception as e:
                self.logger.error(f"监控循环出错：{str(e)}")
    
    def update_progress(self, task_id: str, role_name: str, status: str, progress: str):
        """更新任务进度"""
        progress_info = TaskProgress(
            task_id=task_id,
            role_name=role_name,
            status=status,
            progress=progress,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.task_progress[task_id] = progress_info
        self.progress_queue.put(progress_info)
    
    def _report_progress(self, progress: TaskProgress):
        """汇报任务进度"""
        status_emoji = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌"
        }
        emoji = status_emoji.get(progress.status, "❓")
        print(f"\n{emoji} 任务进展 [{progress.timestamp}]")
        print(f"   任务ID: {progress.task_id}")
        print(f"   角色: {progress.role_name}")
        print(f"   状态: {progress.status}")
        print(f"   进度: {progress.progress}")
    
    def get_task_status(self, task_id: str) -> TaskProgress:
        """获取任务状态"""
        return self.task_progress.get(task_id)

class MainDialogUnit:
    """主对话单元 - 持续接收用户输入和交流"""
    def __init__(self, agent_studio: 'AgentStudio'):
        self.agent_studio = agent_studio
        self.logger = logging.getLogger("主对话单元")
        self.task_monitor = TaskMonitor()
        self.current_task_thread = None
        self.task_running = False
        self.task_results = {}
        
        # 运行时模型名称
        self.dialog_model_name = "main-dialog-unit"  # 主对话单元专用模型
        # 直接使用 AgentStudio 创建的意图识别模型，避免重复创建
        self.intent_model_name = agent_studio.intent_model_name
        
        # 创建运行时模型（只创建主对话单元模型，不重复创建意图识别模型）
        self._create_runtime_models()
    
    def _create_runtime_models(self):
        """创建运行时专用模型"""
        try:
            # 创建主对话单元专用模型
            self.logger.info(f"正在创建主对话单元专用模型：{self.dialog_model_name}")
            ollama.create(
                model=self.dialog_model_name,
                from_=MODEL_EXEC_CHAT,
                system="你是智能体工作室的主对话单元，负责与用户进行持续的对话交流。你的职责是：\n1. 友好、自然地回答用户的简单问题和闲聊\n2. 保持对话的连贯性和互动性\n3. 在用户查询任务状态时，提供友好的回应\n4. 保持专业但亲切的语气\n\n请始终以智能体工作室助手的身份回应，不要自称其他身份。"
            )
            self.logger.info(f"主对话单元专用模型创建成功：{self.dialog_model_name}")
            
        except Exception as e:
            self.logger.error(f"创建运行时模型失败：{str(e)}")
            # 如果创建失败，使用原模型作为后备
            self.dialog_model_name = MODEL_EXEC_CHAT
    
    def _cleanup_runtime_models(self):
        """清理运行时专用模型"""
        try:
            # 删除主对话单元专用模型
            if self.dialog_model_name != MODEL_EXEC_CHAT:
                self.logger.info(f"正在删除主对话单元专用模型：{self.dialog_model_name}")
                ollama.delete(self.dialog_model_name)
                self.logger.info(f"主对话单元专用模型删除成功：{self.dialog_model_name}")
        except Exception as e:
            self.logger.error(f"删除运行时模型失败：{str(e)}")
    
    def start(self):
        """启动主对话单元"""
        self.task_monitor.start_monitoring()
        self.logger.info("主对话单元已启动")
        self._main_loop()
    
    def stop(self):
        """停止主对话单元"""
        self.task_monitor.stop_monitoring()
        self._cleanup_runtime_models()
        # 清理AgentStudio创建的运行时模型
        self.agent_studio._cleanup_runtime_models()
        self.logger.info("主对话单元已停止")
    
    def _main_loop(self):
        """主循环"""
        print("=== 智能体工作室 ===")
        print("主对话单元已启动，可以随时与我交流")
        print("输入 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n您: ")
                if user_input.lower() == 'exit':
                    print("正在退出智能体工作室...")
                    self.stop()
                    break
                
                # 处理用户输入
                self._handle_user_input(user_input)
                
            except KeyboardInterrupt:
                print("\n检测到中断信号，正在退出...")
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"处理用户输入时出错：{str(e)}")
                print(f"抱歉，处理您的请求时出错：{str(e)}")
    
    def _handle_user_input(self, user_input: str):
        """处理用户输入"""
        # 识别用户意图
        intent = self._identify_intent(user_input)
        
        # 如果有任务在运行，检查是否是查询任务状态或对话
        if self.task_running:
            if intent == "MONITOR":
                # 先答复用户，然后再查询任务状态
                print("\n智能体: 我已经让人去查了，请稍候...")
                # 模拟查询过程的延迟
                time.sleep(1)
                # 再通过监控单元获取详细的任务进展
                self._report_task_status()
            elif intent == "CHAT":
                # 任务运行时，仍然可以对话
                self._handle_chat(user_input)
            else:
                # 任务运行时，不允许启动新任务
                print("\n⚠️  当前已有任务在运行，请等待当前任务完成后再提交新任务")
        else:
            # 没有任务运行，根据意图类型处理
            if intent == "MONITOR":
                # 先答复用户，然后再查询任务状态
                print("\n智能体: 我已经让人去查了，请稍候...")
                # 模拟查询过程的延迟
                time.sleep(1)
                # 再通过监控单元获取详细的任务进展
                self._report_task_status()
            elif intent == "TASK":
                # 启动任务执行
                self._start_task(user_input)
            elif intent == "CHAT":
                # 简单对话
                self._handle_chat(user_input)
            else:
                # 默认按对话处理
                self._handle_chat(user_input)
    
    def _identify_intent(self, user_input: str) -> str:
        """识别用户意图（使用模型语义识别）"""
        try:
            # 使用意图识别单元专用模型进行意图识别
            response = ollama.chat(
                model=self.intent_model_name,  # 使用意图识别单元专用模型
                messages=[
                    {"role": "user", "content": user_input}
                ],
                options={"temperature": 0.0, "num_ctx": 2048}
            )
            
            intent = response['message']['content'].strip()
            
            # 确保返回有效的意图类型
            if intent in ["TASK", "CHAT", "MONITOR"]:
                return intent
            else:
                return "CHAT"
        except Exception as e:
            # 出错时默认按对话处理
            return "CHAT"
    
    def _report_task_status(self):
        """汇报任务状态"""
        # 从任务监控单元获取任务状态信息
        if not self.task_monitor.task_progress:
            print("\n📊 当前没有正在运行的任务")
            return
        
        print("\n📊 当前任务状态：")
        print("=" * 50)
        for task_id, progress in self.task_monitor.task_progress.items():
            status_emoji = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }
            emoji = status_emoji.get(progress.status, "❓")
            print(f"{emoji} {progress.role_name} ({progress.task_id})")
            print(f"   状态: {progress.status}")
            print(f"   进度: {progress.progress}")
            print(f"   时间: {progress.timestamp}")
            print("-" * 50)
    
    def _handle_chat(self, user_input: str):
        """处理简单对话"""
        try:
            self.logger.info(f"处理对话：{user_input}")
            # 使用主对话单元专用模型处理简单对话，开启流输出
            print("\n智能体: ", end="")
            
            # 用于存储完整的响应
            full_response = ""
            
            # 开启流输出
            stream = ollama.chat(
                model=self.dialog_model_name,  # 使用主对话单元专用模型
                messages=[
                    {"role": "user", "content": user_input}
                ],
                stream=True,
                options={"temperature": 0.7, "num_ctx": 4096}
            )
            
            # 处理流输出
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            # 确保有换行
            print()
            
            self.logger.info("简单对话处理完成")
        except Exception as e:
            self.logger.error(f"对话处理失败：{str(e)}")
            error_message = "抱歉，我暂时无法回答您的问题。"
            print(f"\n智能体: {error_message}")
    
    def _start_task(self, user_input: str):
        """启动任务执行"""
        if self.task_running:
            print("\n⚠️  当前已有任务在运行，请等待当前任务完成后再提交新任务")
            return
        
        self.task_running = True
        self.task_results = {}
        self.task_progress = {}
        
        # 清空任务监控单元的历史状态
        self.task_monitor.task_progress = {}
        
        # 设置AgentStudio的任务监控和回调函数
        self.agent_studio.set_task_monitor(self.task_monitor)
        self.agent_studio.set_task_progress_callback(self.task_monitor.update_progress)
        
        # 在新线程中执行任务，避免阻塞主对话
        self.current_task_thread = threading.Thread(
            target=self._execute_task,
            args=(user_input,),
            daemon=True
        )
        self.current_task_thread.start()
        
        print("\n🚀 任务已启动，正在后台执行...")
        print("💡 您可以继续与我交流，或输入'任务状态'查看进度")
    
    def _execute_task(self, user_input: str):
        """执行任务（在独立线程中）"""
        try:
            # 更新任务状态为运行中
            self.task_monitor.update_progress("main", "任务执行", "running", "任务开始执行")
            
            # 执行任务
            result = self.agent_studio.run(user_input)
            
            # 任务完成
            self.task_results["main"] = result
            self.task_monitor.update_progress("main", "任务执行", "completed", "任务执行完成")
            
            # 汇报最终结果
            print("\n" + "=" * 50)
            print("📋 任务执行结果")
            print("=" * 50)
            print(result)
            print("=" * 50)
            
        except Exception as e:
            self.logger.error(f"任务执行失败：{str(e)}")
            self.task_monitor.update_progress("main", "任务执行", "failed", f"任务执行失败：{str(e)}")
            print(f"\n❌ 任务执行失败：{str(e)}")
        finally:
            self.task_running = False

if __name__ == "__main__":
    # 初始化智能体工作室
    agent_studio = AgentStudio()
    
    # 启动主对话单元
    dialog_unit = MainDialogUnit(agent_studio)
    dialog_unit.start()
