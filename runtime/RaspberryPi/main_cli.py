from process import UartListener, DataProcessor, Real_time_Data, MatrixFIFO
from queue import Queue, Empty
import numpy as np
import sys
import time
import os
import threading

# 全局变量
R3_Data = Queue()
collector = None
processor = None
collector1 = None
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

def find_available_port():
    """
    按照优先级查找可用的串口设备
    """
    # 优先使用我们定义的符号链接
    if os.path.exists('/dev/ttyRadar'):
        return '/dev/ttyRadar'
    
    # 其次尝试使用 ttyACM1
    if os.path.exists('/dev/ttyACM1'):
        return '/dev/ttyACM1'
    
    # 最后尝试使用 ttyACM0
    if os.path.exists('/dev/ttyACM0'):
        return '/dev/ttyACM0'
    
    # 如果都不存在，返回默认值
    return '/dev/ttyACM1'

def openradar():
    global collector, processor, collector1
    processor = DataProcessor('Processor', None, matrix_fifo, R3_Data)
    # processor.sendLabel.connect(lambda d: printlogdict(d))
    # processor.setlabelinit()
    
    # 使用优先级方法查找可用串口
    port = find_available_port()
    printlog(f"使用串口设备: {port}", "blue")
    collector = UartListener('Listener', frame_length, port)
    collector1 = Real_time_Data('Real_time_Data', matrix_fifo)
    
    collector.start()
    processor.start()
    collector1.start()
    
    printlog("雷达数据处理已启动", "green")

def process_data():
    """
    处理数据的主循环
    """
    global running
    
    while running:
        try:
            # 非阻塞地获取数据
            r3_data = R3_Data.get_nowait()
            # 在命令行模式下，我们可以打印一些数据统计信息
            # printlog(f"接收到数据，形状: {r3_data.shape if hasattr(r3_data, 'shape') else 'N/A'}", "green")
        except Empty:
            # 没有数据，等待一会
            time.sleep(0.1)
        except Exception as e:
            printlog(f"数据处理错误: {str(e)}", "red")
            time.sleep(1)

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
    global frame_length, matrix_fifo, running
    
    fft_sample = 151
    chirp = 1
    tx_num = 1
    rx_num = 4
    header_size = 3
    end_size = 4
    frame_length = (fft_sample * chirp * 2 + header_size + end_size) * tx_num * rx_num
    frame_length = 1 * 128 * 2 + header_size + end_size
    matrix_fifo = MatrixFIFO(num_chirps=64, num_adcsamples=32, max_num_chirps=1024*4)
    
    printlog("雷达系统初始化完成", "blue")
    printlog("=== 无界面版雷达系统 ===", "green")
    
    # 创建线程等待命令
    command_thread = threading.Thread(target=wait_for_command)
    command_thread.daemon = True
    command_thread.start()
    
    # 创建数据处理线程
    data_thread = threading.Thread(target=process_data)
    data_thread.daemon = True
    data_thread.start()
    
    try:
        # 主线程等待，直到用户终止程序
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        printlog("用户中断，程序退出", "red")
    finally:
        global collector, processor, collector1
        running = False
        
        # 清理资源
        if collector is not None and collector.isRunning():
            collector.terminate()
        if processor is not None and processor.isRunning():
            processor.terminate()
        if collector1 is not None and collector1.isRunning():
            collector1.terminate()
        
        printlog("程序已关闭", "blue")

if __name__ == '__main__':
    import os
    print(f"My app PID: {os.getpid()}")
    main()
