import numpy as np
import os
# import threading as th
from collections import deque
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from helpers.DopplerAlgo import *
import datetime
import leapcdll.leap as leap
from leapcdll.leap import datatypes as ldt
from common_words import common_words
from PyQt5.QtCore import QThread, pyqtSignal

#存储数据从float64改成float32, 

#前n次get到空还是保留返回最后一次get到的数据，一旦get了n次还是空就返回[np.nan, np.nan, 1]
class DequeFIFO:
    def __init__(self, n):
        self.fifo = deque(maxlen=1)
        self.empty_gets = 0  # 计数器，用于记录连续空get调用的次数
        self.last_non_empty = [np.nan, np.nan, 1]  # 存储最后一次非空的数据
        self.n = n  # 设置允许的最大空get次数
    
    def set(self, value):
        # 添加元素到deque，超出长度的旧元素会被自动丢弃
        self.fifo.append(value)
        self.empty_gets = 0  # 重置空get计数器
        self.last_non_empty = value  # 更新最后一次非空的数据
        # print(f"Set: {value}")
    
    def get(self):
        if self.fifo:
            item = self.fifo.popleft()  # 移除并返回最左端元素
            self.last_non_empty = item  # 更新最后一次非空的数据
            self.empty_gets = 0  # 重置空get计数器            
            # print(f"Got: {item}")
            return item
        else:
            self.empty_gets += 1  # 增加空get计数器
            if self.empty_gets >= self.n:
                # print(f"Returning: [np.nan, np.nan, 1]")
                return [np.nan, np.nan, 1]  # 如果空get次数达到n次，返回指定数组
            else:
                # print(f"Returning last non-empty data or initial value: {self.last_non_empty}")
                return self.last_non_empty  # 否则返回最后一次非空的数据或初始化值

class MatrixFIFO:
    def __init__(self, num_chirps=64, num_adcsamples=64, max_num_chirps=1024):
        self.num_chirps = num_chirps
        self.num_adcsamples = num_adcsamples
        self.max_num_chirps = max_num_chirps   
        self.current_matrix = np.empty((0, num_adcsamples), dtype=np.float32)  # 初始化为空矩阵

    def append_matrix(self, new_matrix):
        if new_matrix.shape != (self.num_chirps, self.num_adcsamples):  # 检查新矩阵尺寸
            raise ValueError("New matrix must be of shape (self.num_chirp, self.num_adcsample")

        # 沿第二维拼接新矩阵
        self.current_matrix = np.concatenate((self.current_matrix, new_matrix), axis=0)

        # 如果当前矩阵行数超过最大行数，则移除最早的行
        if self.current_matrix.shape[0] > self.max_num_chirps:
            # print("满了")
            # 计算超出的行数
            excess_rows = self.current_matrix.shape[0] - self.max_num_chirps
            # 移除最早的行
            self.current_matrix = self.current_matrix[excess_rows:, :]

    def remove_rows(self, num_rows):
        # 确保不会尝试移除超过当前矩阵行数的行
        if num_rows > self.current_matrix.shape[0]:
            raise ValueError("Attempting to remove more rows than currently exist.")
        
        # 移除指定数量的列
        self.current_matrix = self.current_matrix[num_rows:, :]

    def get_matrix(self):
        return self.current_matrix
    
    def get_matrix_rows(self, rows):
        return self.current_matrix[:rows, :] 
    
class PinchingListener(leap.Listener):

    def __init__(self, indexPosition):
        super().__init__()
        self.indexPosition = indexPosition
    def location_end_of_finger(self, hand: ldt.Hand, digit_idx: int) -> ldt.Vector:
        digit = hand.digits[digit_idx]
        return digit.distal.next_joint


    def sub_vectors(self, v1: ldt.Vector, v2: ldt.Vector) -> list:
        return map(float.__sub__, v1, v2)


    def fingers_pinching(self, thumb: ldt.Vector, index: ldt.Vector):
        diff = list(map(abs, self.sub_vectors(thumb, index)))

        if diff[0] < 40 and diff[1] < 40 and diff[2] < 40:
            return True, diff
        else:
            return False, diff
        
    def on_connection_event(self, event):
        pass
        # print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        # print(f"Found device {info.serial}")
        
    def on_tracking_event(self, event):
        # global indexPosition
        # if event.tracking_frame_id % 1 == 0:
        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            if hand_type == "Right":
                index = self.location_end_of_finger(hand, 1) # 食指
                middle = self.location_end_of_finger(hand, 2) # 中指
                # print("{global is_draw_flag}")
                # pinching_str = "not pinching" if not pinching else "" + str("pinching")
                # print(
                #     f"{hand_type} hand index and middle {pinching_str} with position diff ({array[0]}, {array[1]}, {array[2]})."
                # )
                is_draw_flag, array = self.fingers_pinching(index, middle)
                self.indexPosition.set([index[0],index[1],int(is_draw_flag)])

class UartListener(QThread):
    def __init__(self, name, indexPosition, matrix_fifo):
        QThread.__init__(self)
        self.indexPosition = indexPosition
        self.matrix_fifo = matrix_fifo
        self.RadarConfig = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.0384024567902088178,  # Frame repetition time 0.15s (frame rate of 6.667Hz)
        chirp_repetition_time_s=0.0006000000284984708,  # Chirp repetition time (or pulse repetition time) of 0.5ms
        num_chirps=64,  # chirps per frame
        tdm_mimo=False,  # MIMO disabled
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=58e9,  # start frequency: 60 GHz
            end_frequency_Hz=63e9,  # end frequency: 61.5 GHz
            sample_rate_Hz=500000,  # ADC sample rate of 1MHz
            num_samples=32,  # 64 samples per chirp
            rx_mask=1,  # RX antennas 1 and 3 activated
            tx_mask=1,  # TX antenna 1 activated
            tx_power_level=31,  # TX power level of 31
            lp_cutoff_Hz=500000,  # Anti-aliasing cutoff frequency of 500kHz
            hp_cutoff_Hz=80000,  # 80kHz cutoff frequency for high-pass filter
            # 高通滤波如果比较低会出现对称情况
            if_gain_dB=28,  # 33dB if gain
            )
        )

    def run(self):
    # 初始化leap motion
        listener = PinchingListener(self.indexPosition)
        connection = leap.Connection()
        connection.add_listener(listener)
        connection.set_tracking_mode(leap.TrackingMode.Desktop)# 默认

        with DeviceFmcw() as device, connection.open():

            sequence = device.create_simple_sequence(self.RadarConfig)
            device.set_acquisition_sequence(sequence)

            # Fetch a number of frames
            while(True):
                frame_contents = device.get_next_frame()

                for frame in frame_contents:
                    self.matrix_fifo.append_matrix(frame[0, :, :])
    
class DataProcessor(QThread):
    sendLabel = pyqtSignal('PyQt_PyObject')

    def __init__(self, name, indexPosition, matrix_fifo, r1_data, r3_data, r4_data):
        QThread.__init__(self)
        self.r1_data = r1_data
        self.r3_data = r3_data
        self.r4_data = r4_data
        self.num_2dfft = 128
        self.doppler = DopplerAlgo(32, self.num_2dfft, num_ant=1)
        self.indexPosition = indexPosition
        self.matrix_fifo = matrix_fifo
        self.length_max = 1024
        self.udoppler_queue = deque(maxlen=self.length_max)
        self.range_queue = deque(maxlen=self.length_max)
        self.xzflag_queue = deque(maxlen=self.length_max)
        self.count = 0 #用于计算保存多少帧的
        # 指定需要检查和创建的文件夹名称
        self.folder_name = 'datas4'
        # 检查文件夹是否存在
        if not os.path.exists(self.folder_name):
            # 如果文件夹不存在，则创建它
            os.makedirs(self.folder_name)
        # 读取文件夹中所有的 .npy 文件
        # self.file_number = len([f for f in os.listdir(self.folder_name) if f.endswith('.npy')])
        # 打开 txt 文件用于记录
        self.log_file = open(f"{self.folder_name}/file_log.txt", 'a')

    def setlabelinit(self):
        self.label = common_words[np.random.randint(0, len(common_words))]
        self.sendinfo(self.label,"blue")
    
    def sendinfo(self, string, fontcolor):
        d = dict()
        d["string"] = "请写入:"+string
        d["fontcolor"] = fontcolor
        self.sendLabel.emit(d)

    def run(self):
        with self.log_file:
            while(True):
                sizematrix = self.matrix_fifo.get_matrix().shape
                if sizematrix[0]>=self.num_2dfft:
                    frame = self.matrix_fifo.get_matrix_rows(self.num_2dfft)
                    
                    # 由于这里的导致数据不是实时的
                    self.matrix_fifo.remove_rows(16)
                    rd_spectrum = self.doppler.compute_doppler_map(frame, 0)
                    
                    udoppler1D = np.sum(np.abs(rd_spectrum),axis=0, dtype=np.float32)
                    range1D = np.sum(np.abs(rd_spectrum),axis=1, dtype=np.float32)
                    # udoppler1D = np.zeros((256,1))
                    # range1D = np.zeros((256,1))
                    self.udoppler_queue.append(udoppler1D)
                    
                    self.range_queue.append(range1D)
                    item = self.indexPosition.get()
                    self.xzflag_queue.append(item)

                    if(item[0] is not np.nan):
                        # pass
                        self.count += 1
                    else:
                        if self.count:
                            if self.count > self.length_max:
                                print("保存数据过长将被截断")
                            length = min(self.count, self.length_max)
                            saveudoppler = np.array(self.udoppler_queue)[-length:-1,:]
                            
                            saverange = np.array(self.range_queue)[-length:-1,:]
                            savexzflag = np.array(self.xzflag_queue, dtype=np.float32)[-length:-1,:]# 不要最后一行，因为是nan来的

                            savedata = np.concatenate((saveudoppler,saverange,savexzflag), axis=1)
                            # 构建文件名
                            # 获取当前的日期和时间
                            current_time = datetime.datetime.now()

                            # 格式化日期时间字符串，例如 '20240503_153045' （年月日_时分秒）
                            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

                            # 创建文件名，这里假设 folder_name 已经在类的其他部分被定义
                            filename = f"{self.folder_name}/data_{formatted_time}.npy"
                            np.save(filename, savedata)
                            # 将文件名和标签写入 txt 文件
                            # keys, 相对路径, 长度, label
                            self.log_file.write(f"{formatted_time}\t{filename}\t{length-1}\t{self.label}\n")
                            self.log_file.flush()  # 确保即时写入到文件
                            # print("保存数据")
                            self.label = common_words[np.random.randint(0, len(common_words))]
                            self.sendinfo(formatted_time+':'+self.label,"blue")
                            # self.file_number += 1
                            self.count = 0



                    self.r1_data.put(np.array(self.xzflag_queue))
                    self.r3_data.put(np.array(self.udoppler_queue))
                    self.r4_data.put(np.array(self.range_queue))
                

