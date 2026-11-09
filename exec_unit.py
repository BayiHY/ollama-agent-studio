import ollama
import logging
import io
from contextlib import redirect_stdout
from models import TaskRole, ExecResult

class ExecUnit:
    """执行单元（动态创建，无数量限制）- 无思考模式，按要求执行"""
    def __init__(self, task_role: TaskRole, model: str, progress_callback=None):
        self.task_role = task_role  # 子任务+角色信息
        self.base_model = model  # 基础模型
        self.task_id = task_role.task_id
        self.role_name = task_role.role_name
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(f"执行单元-{self.role_name}")
        
        # 生成唯一模型名称
        self.model_name = f"exec-{self.task_id}-{self.role_name.lower().replace(' ', '-')}"
        self.created_model = False
        
        # 动态创建模型（基于角色定义）
        self._create_model()
    
    def _create_model(self):
        """基于角色定义创建专用模型"""
        try:
            # 构建system prompt
            system_prompt = "你的唯一身份是：" + self.role_name + "\n你的核心职责是：" + self.task_role.role_duty + "\n你的执行方式是：" + self.task_role.execute_way
            
            # 使用create方法创建模型
            ollama.create(
                model=self.model_name,
                from_=self.base_model,
                system=system_prompt,
                parameters={"temperature": 0.0, "num_ctx": 4096}
            )
            self.created_model = True
            self.logger.info(f"成功创建模型：源模型='{self.base_model}' -> 目标模型='{self.model_name}'")
        except Exception as e:
            self.logger.error(f"创建模型时出错：源模型='{self.base_model}' -> 目标模型='{self.model_name}'，错误信息='{str(e)}'，将使用基础模型")
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
                
                # 提取结果
                result = response['message']['content']
            
            self.logger.info("子任务执行完成")
            
            # 汇报任务完成
            if self.progress_callback:
                self.progress_callback(self.task_id, self.role_name, "completed", "任务执行完成")
            
            # 返回执行结果
            return ExecResult(
                task_id=self.task_id,
                role_name=self.role_name,
                model_used=model_to_use,
                result=result,
                success=True
            )
        except Exception as e:
            self.logger.error("执行子任务时出错：" + str(e))
            
            # 汇报任务失败
            if self.progress_callback:
                self.progress_callback(self.task_id, self.role_name, "failed", "任务执行失败：" + str(e))
            
            # 返回失败结果
            return ExecResult(
                task_id=self.task_id,
                role_name=self.role_name,
                model_used=model_to_use,
                result="执行失败：" + str(e),
                success=False
            )
        finally:
            # 任务执行完成后删除模型
            self._cleanup_model()
    
    def _execute_math_task(self, model: str, task_input: str) -> str:
        """执行数学计算任务（生成代码并执行）"""
        # 构建计算任务prompt
        math_prompt = task_input + "\n\n请生成Python代码来解决这个数学问题，代码应该：\n1. 直接计算结果\n2. 不要使用任何外部库\n3. 最后使用print()函数输出计算结果\n4. 只输出代码，不输出其他内容"
        
        try:
            # 获取模型生成的代码
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "user", "content": math_prompt}
                ],
                options={"temperature": 0.0, "num_ctx": 4096}
            )
            
            code = response['message']['content']
            self.logger.info("生成的计算代码：" + code)
            
            # 执行代码并捕获输出
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code)
            output = f.getvalue().strip()
            
            self.logger.info("计算结果：" + output)
            return output
        except Exception as e:
            self.logger.error("执行计算任务时出错：" + str(e))
            raise
    
    def _cleanup_model(self):
        """任务执行完成后清理模型"""
        if self.created_model:
            try:
                self.logger.info("开始删除模型：" + self.model_name)
                ollama.delete(self.model_name)
                self.logger.info("成功删除模型：" + self.model_name)
            except Exception as e:
                self.logger.error("删除模型时出错：" + str(e))
