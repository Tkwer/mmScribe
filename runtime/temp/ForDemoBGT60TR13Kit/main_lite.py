from process_lite import UartListener, DataProcessor, MatrixFIFO
from queue import Queue, Empty
import numpy as np
import sys
import time
import os
import threading

# 全局变量
R1_Data = Queue()
R3_Data = Queue()
R4_Data = Queue()
collector = None
processor = None
running = True

def printlog(message, color="white"):
    """
    在控制台打印日志信息
    """
    gettime = time.strftime("%H:%M:%S", time.localtime())
    if color == "blue":
        # 蓝色
        print(f"\033[94m{gettime} --> {message}\033[0m")
    elif color == "red":
        # 红色
        print(f"\033[91m{gettime} --> {message}\033[0m")
    elif color == "green":
        # 绿色
        print(f"\033[92m{gettime} --> {message}\033[0m")
    else:
        # 默认白色
        print(f"{gettime} --> {message}")

def printlogdict(d):
    string = d["string"]
    fontcolor = d["fontcolor"]
    printlog(string, fontcolor)

class DummyLogWidget:
    """模拟GUI日志窗口的类，用于传递给process_lite中需要接收信号的函数"""
    def __init__(self):
        pass
    
    def moveCursor(self, *args):
        pass
    
    def append(self, text):
        # 移除HTML标签，只保留文本内容
        import re
        clean_text = re.sub(r'<.*?>', '', text)
        print(clean_text)

def openradar():
    global collector, processor
    printlog('开始雷达采集和处理', 'green')
    
    dummy_log = DummyLogWidget()
    collector = UartListener('Listener', indexPosition, matrix_fifo)
    processor = DataProcessor('Processor', indexPosition, matrix_fifo, R1_Data, R3_Data, R4_Data)
    
    # 连接信号到自定义的处理函数
    processor.sendLabel.connect(lambda d: printlogdict(d))
    processor.setlabelinit()
    
    collector.start()
    processor.start()

def wait_for_command():
    """
    等待用户输入命令
    """
    global running
    
    while running:
        cmd = input("输入 'y' 启动雷达，输入 'q' 退出: ")
        if cmd.lower() == 'y':
            printlog("启动雷达数据处理...", "blue")
            openradar()
            break
        elif cmd.lower() == 'q':
            running = False
            printlog("程序退出", "red")
            break
        else:
            printlog("无效命令，请输入 'y' 启动或 'q' 退出", "red")

def main():
    # 雷达配置
    global indexPosition, matrix_fifo, running
    
    # 创建FIFO实例
    indexPosition = None
    matrix_fifo = MatrixFIFO(num_chirps=64, num_adcsamples=32, max_num_chirps=1024)
    
    printlog("雷达系统初始化完成", "blue")
    printlog("=== 无界面版雷达系统 ===", "green")
    
    # 创建线程等待命令
    command_thread = threading.Thread(target=wait_for_command)
    command_thread.daemon = True
    command_thread.start()

    
    try:
        # 主线程等待，直到用户终止程序
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        printlog("用户中断，程序退出", "red")
    finally:
        global collector, processor
        running = False
        
        # 清理资源
        if collector is not None and collector.isRunning():
            collector.terminate()
        if processor is not None and processor.isRunning():
            processor.terminate()
        
        printlog("程序已关闭", "blue")

if __name__ == '__main__':
    import os
    print(f"程序PID: {os.getpid()}")
    main()





