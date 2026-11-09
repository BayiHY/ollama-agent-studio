import ollama
import logging
import threading
import time
from task_monitor import TaskMonitor

class MainDialogUnit:
    """主对话单元 - 持续接收用户输入和交流"""
    def __init__(self, agent_studio):
        self.agent_studio = agent_studio
        self.logger = logging.getLogger("主对话单元")
        self.task_monitor = TaskMonitor()
        self.current_task_thread = None
        self.task_running = False
        self.task_results = {}
        
        # 运行时模型名称
        self.dialog_model_name = "main-dialog-unit"  # 主对话单元专用模型
        self.intent_model_name = "intent-recognition-unit"  # 意图识别单元专用模型
        
        # 创建运行时模型
        self._create_runtime_models()
    
    def _create_runtime_models(self):
        """创建运行时专用模型"""
        try:
            from constants import MODEL_EXEC_CHAT, MODEL_INTENT
            
            # 创建主对话单元专用模型
            self.logger.info(f"正在创建主对话单元专用模型：源模型='{MODEL_EXEC_CHAT}' -> 目标模型='{self.dialog_model_name}'")
            ollama.create(
                model=self.dialog_model_name,
                from_=MODEL_EXEC_CHAT,
                system="你是智能体工作室的主对话单元，负责与用户进行持续的对话交流。你的职责是：\n1. 友好、自然地回答用户的简单问题和闲聊\n2. 保持对话的连贯性和互动性\n3. 在用户查询任务状态时，提供友好的回应\n4. 保持专业但亲切的语气\n\n请始终以智能体工作室助手的身份回应，不要自称其他身份。"
            )
            self.logger.info(f"主对话单元专用模型创建成功：源模型='{MODEL_EXEC_CHAT}' -> 目标模型='{self.dialog_model_name}'")
            
            # 创建意图识别单元专用模型
            self.logger.info(f"正在创建意图识别单元专用模型：源模型='{MODEL_INTENT}' -> 目标模型='{self.intent_model_name}'")
            ollama.create(
                model=self.intent_model_name,
                from_=MODEL_INTENT,
                system="你是智能体工作室的意图识别单元，负责判断用户输入的意图。你的职责是：\n1. 准确识别用户是要执行任务、简单对话还是查询任务监控\n2. 只输出\"TASK\"、\"CHAT\"或\"MONITOR\"中的一个词\n3. 不要输出任何其他内容或解释\n\n判断标准：\n- TASK：用户要求执行具体任务（如计算、创作、分析、编程、翻译等）\n- CHAT：用户进行简单对话（如问候、闲聊、咨询、感谢等）\n- MONITOR：用户查询任务监控信息（如询问任务进度、状态、结果等）\n\n请只输出一个词，不要输出其他内容。"
            )
            self.logger.info(f"意图识别单元专用模型创建成功：源模型='{MODEL_INTENT}' -> 目标模型='{self.intent_model_name}'")
            
        except Exception as e:
            self.logger.error(f"创建运行时模型失败：错误信息='{str(e)}'")
            # 如果创建失败，使用原模型作为后备
            from constants import MODEL_EXEC_CHAT, MODEL_INTENT
            self.dialog_model_name = MODEL_EXEC_CHAT
            self.intent_model_name = MODEL_INTENT
    
    def _cleanup_runtime_models(self):
        """清理运行时专用模型"""
        try:
            from constants import MODEL_EXEC_CHAT, MODEL_INTENT
            
            # 删除主对话单元专用模型
            if self.dialog_model_name != MODEL_EXEC_CHAT:
                self.logger.info(f"正在删除主对话单元专用模型：{self.dialog_model_name}")
                ollama.delete(self.dialog_model_name)
                self.logger.info(f"主对话单元专用模型删除成功：{self.dialog_model_name}")
            
            # 删除意图识别单元专用模型
            if self.intent_model_name != MODEL_INTENT:
                self.logger.info(f"正在删除意图识别单元专用模型：{self.intent_model_name}")
                ollama.delete(self.intent_model_name)
                self.logger.info(f"意图识别单元专用模型删除成功：{self.intent_model_name}")
        except Exception as e:
            self.logger.error(f"删除运行时模型失败：{str(e)}")
    
    def start(self):
        """启动主对话单元"""
        self.task_monitor.start_monitoring()
        # 设置AgentStudio的任务监控和回调函数
        self.agent_studio.set_task_monitor(self.task_monitor)
        self.agent_studio.set_task_progress_callback(self.task_monitor.update_progress)
        self.logger.info("主对话单元已启动")
        self._main_loop()
    
    def stop(self):
        """停止主对话单元"""
        self.task_monitor.stop_monitoring()
        self._cleanup_runtime_models()
        # 停止智能体工作室，清理运行时模型
        self.agent_studio.stop()
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
