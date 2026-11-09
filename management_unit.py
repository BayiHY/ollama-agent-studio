import ollama
import json
import re
import logging
from typing import List
from models import TaskRole, ExecResult

class ManagementUnit:
    """管理控制单元（qwen3:1.7b）- 无思考模式，固定流程执行"""
    def __init__(self, model_name: str):
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
                # 解析JSON
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
        from constants import MODEL_EXEC_ADVANCED, MODEL_EXEC_ADVANCED_CODER, MODEL_EXEC_CORE, MODEL_EXEC_QUICK_CODER, MODEL_EXEC_LIGHT, MODEL_EXEC_CHAT, MODEL_EXEC_SIMPLE, MODEL_EXEC_MINIMAL_CODER
        
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
