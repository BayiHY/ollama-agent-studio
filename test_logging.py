import os
import logging

# 测试日志功能
def test_logging():
    # 使用绝对路径确保日志文件在正确的位置
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_log.log")
    print(f"测试日志文件路径：{log_file_path}")
    
    # 手动配置日志
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
    
    # 获取测试日志器
    logger = logging.getLogger("测试日志")
    logger.info("测试日志开始")
    logger.info("这是一条测试日志")
    logger.info("测试日志结束")
    
    print("测试日志写入完成")
    
    # 检查文件是否存在
    if os.path.exists(log_file_path):
        print(f"日志文件已创建：{log_file_path}")
        # 读取文件内容
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"日志文件内容：{content}")
        except Exception as e:
            print(f"读取日志文件失败：{e}")
    else:
        print("日志文件未创建")

if __name__ == "__main__":
    test_logging()
