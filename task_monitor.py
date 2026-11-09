import logging
import threading
import time
from queue import Queue
from models import TaskProgress

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
