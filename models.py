from dataclasses import dataclass
from typing import List

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
    complexity: str            # 复杂度（用于匹配模型）
    dependencies: List[str]     # 依赖的其他任务ID

# 数据类：执行结果（结构化，便于汇总）
@dataclass
class ExecResult:
    task_id: str     # 子任务ID
    role_name: str     # 执行角色
    model_used: str    # 使用的Ollama模型
    result: str        # 子任务执行结果
    success: bool      # 是否执行成功

# 数据类：任务进度信息
@dataclass
class TaskProgress:
    task_id: str
    role_name: str
    status: str  # pending, running, completed, failed
    progress: str
    timestamp: str
