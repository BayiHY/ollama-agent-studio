import logging
import ollama
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set, Optional, Callable
from constants import MODEL_MANAGER, MODEL_EXEC_ADVANCED, MODEL_EXEC_ADVANCED_CODER, MODEL_EXEC_CORE, MODEL_EXEC_QUICK_CODER, MODEL_EXEC_LIGHT, MODEL_EXEC_CHAT, MODEL_EXEC_SIMPLE, MODEL_EXEC_MINIMAL_CODER, MODEL_INTENT
from models import TaskRole, ExecResult
from management_unit import ManagementUnit
from exec_unit import ExecUnit
from task_monitor import TaskMonitor
from main_dialog_unit import MainDialogUnit

class AgentStudio:
    """智能体工作室 - 统一调度管理"""
    def __init__(self):
        # 日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("agent_studio.log", encoding="utf-8")
            ]
        )
        self.logger = logging.getLogger("智能体工作室")
        
        # 运行时模型名称
        self.management_model_name = "management-control-unit"  # 管理单元专用模型
        
        # 创建运行时模型
        self._create_runtime_models()
        
        # 初始化管理单元
        self.management_unit = ManagementUnit(self.management_model_name)
        
        # 任务监控相关
        self.task_monitor = None
        self.task_progress_callback = None
    
    def _create_runtime_models(self):
        """创建运行时专用模型"""
        try:
            # 创建管理单元专用模型
            self.logger.info(f"正在创建管理单元专用模型：源模型='{MODEL_MANAGER}' -> 目标模型='{self.management_model_name}'")
            ollama.create(
                model=self.management_model_name,
                from_=MODEL_MANAGER,
                system="你是智能体工作室的管理控制单元，负责任务拆解和角色定义。你的职责是：\n1. 接收用户的任务要求，分析任务的复杂度和依赖关系\n2. 将任务拆解为多个子任务，每个子任务对应一个执行单元\n3. 为每个执行单元定义角色名称、职责、执行方式和输出格式\n4. 确定任务之间的依赖关系，确保执行顺序正确\n5. 严格按照JSON格式输出任务规划\n\n请始终保持管理单元的身份，不要自称其他身份。"
            )
            self.logger.info(f"管理单元专用模型创建成功：源模型='{MODEL_MANAGER}' -> 目标模型='{self.management_model_name}'")
            
        except Exception as e:
            self.logger.error(f"创建运行时模型失败：源模型='{MODEL_MANAGER}' -> 目标模型='{self.management_model_name}'，错误信息='{str(e)}'")
            # 如果创建失败，使用原模型作为后备
            self.management_model_name = MODEL_MANAGER
    
    def _cleanup_runtime_models(self):
        """清理运行时专用模型"""
        try:
            # 删除管理单元专用模型
            if self.management_model_name != MODEL_MANAGER:
                self.logger.info(f"正在删除管理单元专用模型：{self.management_model_name}")
                ollama.delete(self.management_model_name)
                self.logger.info(f"管理单元专用模型删除成功：{self.management_model_name}")
        except Exception as e:
            self.logger.error(f"删除运行时模型失败：模型名称='{self.management_model_name}'，错误信息='{str(e)}'")
    
    def set_task_monitor(self, task_monitor: TaskMonitor):
        """设置任务监控单元"""
        self.task_monitor = task_monitor
    
    def set_task_progress_callback(self, callback: Callable):
        """设置任务进度回调函数"""
        self.task_progress_callback = callback
    
    def run(self, user_task: str) -> str:
        """主入口：运行智能体工作室"""
        try:
            # 确保管理单元模型存在
            if self.management_model_name != MODEL_MANAGER:
                # 检查模型是否存在，如果不存在就重新创建
                try:
                    # 尝试使用模型，看看是否存在
                    ollama.chat(
                        model=self.management_model_name,
                        messages=[{"role": "user", "content": "test"}],
                        options={"temperature": 0.0, "num_ctx": 1024}
                    )
                except Exception:
                    # 模型不存在，重新创建
                    self._create_runtime_models()
            
            # 1. 管理单元：任务拆解+角色定义
            self.logger.info("管理单元开始工作：任务拆解与角色定义")
            task_roles = self.management_unit.task_split_and_role_define(user_task)
            
            # 2. 执行单元：并行/串行执行任务
            self.logger.info("执行单元开始工作：并行/串行执行任务")
            exec_results = self._execute_tasks(task_roles)
            
            # 3. 管理单元：结果汇总
            self.logger.info("管理单元开始工作：结果汇总分析")
            final_answer = self.management_unit.summarize_results(exec_results, user_task)
            
            # 注意：不再在run方法中清理运行时模型，而是在stop方法中清理
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"智能体工作室运行失败：{str(e)}")
            return f"智能体工作室运行失败：{str(e)}"
    
    def _execute_tasks(self, task_roles: List[TaskRole]) -> List[ExecResult]:
        """执行所有子任务（支持并行和依赖管理）"""
        # 构建任务依赖图
        task_map: Dict[str, TaskRole] = {}
        dependencies: Dict[str, Set[str]] = {}
        reverse_dependencies: Dict[str, Set[str]] = {}
        
        for task_role in task_roles:
            task_map[task_role.task_id] = task_role
            dependencies[task_role.task_id] = set(task_role.dependencies)
            
            # 构建反向依赖图
            for dep_id in task_role.dependencies:
                if dep_id not in reverse_dependencies:
                    reverse_dependencies[dep_id] = set()
                reverse_dependencies[dep_id].add(task_role.task_id)
        
        # 使用拓扑排序确定执行顺序
        execution_order = self._topological_sort(dependencies)
        
        # 执行任务
        results: List[ExecResult] = []
        result_map: Dict[str, ExecResult] = {}
        
        # 用于跟踪哪些任务已完成
        completed_tasks: Set[str] = set()
        
        # 执行任务
        for task_id in execution_order:
            task_role = task_map[task_id]
            
            # 检查所有依赖是否已完成
            all_deps_completed = all(dep_id in completed_tasks for dep_id in dependencies[task_id])
            
            if all_deps_completed:
                # 执行此任务
                exec_unit = ExecUnit(
                    task_role=task_role,
                    model=self.management_unit.complexity_to_model(task_role.complexity),
                    progress_callback=self.task_progress_callback
                )
                
                # 执行任务
                result = exec_unit.execute_task()
                results.append(result)
                result_map[task_id] = result
                completed_tasks.add(task_id)
            else:
                # 依赖未完成，跳过此任务（应该不会发生，因为拓扑排序已经处理了依赖）
                self.logger.warning(f"任务 {task_id} 的依赖未完成，跳过执行")
        
        return results
    
    def _topological_sort(self, dependencies: Dict[str, Set[str]]) -> List[str]:
        """拓扑排序：确定任务执行顺序"""
        # 复制依赖关系
        deps_copy = {k: set(v) for k, v in dependencies.items()}
        
        # 找出所有没有依赖的任务
        no_deps = [task_id for task_id, deps in deps_copy.items() if not deps]
        
        # 执行拓扑排序
        order: List[str] = []
        
        while no_deps:
            # 取出一个没有依赖的任务
            task_id = no_deps.pop(0)
            order.append(task_id)
            
            # 从其他任务的依赖中移除这个任务
            for other_task in deps_copy:
                if task_id in deps_copy[other_task]:
                    deps_copy[other_task].remove(task_id)
                    # 如果其他任务现在没有依赖了，加入队列
                    if not deps_copy[other_task]:
                        no_deps.append(other_task)
        
        # 检查是否有循环依赖
        if any(deps for deps in deps_copy.values()):
            raise Exception("任务之间存在循环依赖")
        
        return order
    
    def stop(self):
        """停止智能体工作室，清理运行时模型"""
        self._cleanup_runtime_models()
        self.logger.info("智能体工作室已停止并清理所有运行时模型")

if __name__ == "__main__":
    """主入口"""
    try:
        # 初始化智能体工作室
        agent_studio = AgentStudio()
        
        # 启动主对话单元
        main_dialog = MainDialogUnit(agent_studio)
        main_dialog.start()
        
    except Exception as e:
        print(f"启动智能体工作室失败：{str(e)}")
