import os
import numpy as np
from collections import deque
from helpers.DopplerAlgo import *
from PyQt5.QtCore import QThread, pyqtSignal
import ctypes
import time
import datetime
import torch
from typing import Optional, List, Tuple
from recognize_streaming import Streaming_ASRModel, remove_duplicates_and_blank, id_to_char
import logging
from PyQt5.QtCore import QMutex, QMutexLocker  # 使用 Qt 的互斥锁
common_words = [
"the", "of", "and", "a", "to", "in", "is", "you", "that", "it", 
"for", "on", "are", "as", "with", "his", "they", "I", "store","at", 
"this", "have", "from", "or", "one", "had", "by", "word", "but", "not", 
"what", "all", "were", "we", "when", "your", "can", "said", "there", 
"use", "an", "each", "which", "she", "do", "how", "their", "if", "will", 
"up", "other", "about", "out", "many", "then", "them", "these", "so", 
"some", "her", "would", "make", "like", "him", "into", "time", "has", 
"look", "two", "more", "write", "go", "see", "number", "no", "way", 
"could", "people", "my", "than", "first", "water", "been", "call", "who", 
"oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", 
"made", "may", "part", "over", "new", "sound", "take", "only", "little", 
"work", "know", "place", "year", "live", "me", "back", "give", "most", 
"very", "after", "thing", "our", "just", "name", "good", "sentence", 
"man", "think", "say", "great", "where", "help", "through", "much", 
"before", "line", "right", "too", "mean", "old", "any", "same", "tell", 
"boy", "follow", "came", "want", "show", "also", "around", "form", "three", 
"small", "set", "put", "end", "does", "another", "well", "large", "must", 
"big", "even", "such", "because", "turn", "here", "why", "ask", "went", 
"men", "read", "need", "land", "different", "home", "us", "move", "try", 
"kind", "hand", "picture", "again", "change", "off", "play", "spell", 
"air", "away", "animal", "house", "point", "page", "letter", "mother", 
"answer", "found", "study", "still", "learn", "should", "blew", "world",
"high", "every", "near", "add", "food", "between", "own", "below", "country", 
"plant", "last", "school", "father", "keep", "tree", "never", "start", "city", 
"earth", "eye", "light", "thought", "head", "under", "story", "saw", "left", 
"radar", "few", "while", "along", "might", "close", "something", "seem", 
"next", "hard", "open", "example", "begin", "life", "always", "those", "both", 
"paper", "together", "got", "group", "often", "run", "important", "until", 
"children", "side", "feet", "car", "mile", "night", "walk", "white", "sea", 
"began", "grow", "took", "river", "four", "carry", "state", "once", "book", 
"hear", "stop", "without", "second", "later", "miss", "idea", "enough", 
"eat", "face", "watch", "far", "china", "really", "almost", "let", "above", 
"girl", "sometimes", "mountains", "cut", "young", "talk", "soon", "list", 
"song", "being", "leave", "family","be", "he", "signal"]

MAX_ADC_VALUE = 4095.0

class SerialConfig(ctypes.Structure):
    _fields_ = [
        ("portName", ctypes.c_char_p),  # 串口名称
        ("baudRate", ctypes.c_ulong),   # 波特率
        ("byteSize", ctypes.c_ubyte),   # 数据位
        ("stopBits", ctypes.c_ubyte),   # 停止位
        ("parity", ctypes.c_ubyte)      # 奇偶校验
    ]

# 定义雷达帧数据结构体
class RadarFrame(ctypes.Structure):
    _fields_ = [
        ("frameNumber", ctypes.c_uint32),          # 帧序号
        ("frameData", ctypes.POINTER(ctypes.c_uint16)),  # 帧数据，指针到INT16类型数组
        ("chirpNumOneFrame", ctypes.c_uint16),     # 每帧的啁啾数
        ("fft1dNum", ctypes.c_uint16)              # ADC采样点数
    ]

dll = ctypes.cdll.LoadLibrary('dll/uart_capture_rawdata.dll')
# 设置函数原型
dll.openuart.argtypes = [ctypes.POINTER(SerialConfig)]
dll.readData.argtypes = [ctypes.POINTER(RadarFrame)]
dll.sendData.argtypes = [ctypes.c_char_p]
dll.openuart.restype = ctypes.c_int
dll.sendData.restype = ctypes.c_int
dll.readData.restype = ctypes.c_int

# 创建一个实例并初始化
serial_config = SerialConfig()
serial_config.baudRate = 3000000 #2000000   #默认波特率
serial_config.byteSize = 8
serial_config.stopBits = 1
serial_config.parity = 0


radar_frame = RadarFrame()
radar_frame.frameNumber = 0
radar_frame.fft1dNum = 32 #16
radar_frame.chirpNumOneFrame = 64 #32
frame_data_array = np.zeros(radar_frame.fft1dNum*radar_frame.chirpNumOneFrame).astype(ctypes.c_uint16)
radar_frame.frameData = ctypes.cast(frame_data_array.ctypes.data, ctypes.POINTER(ctypes.c_uint16))



# 共用类一定要加锁，不然发生什么错都不知道
class MatrixFIFO:
    def __init__(self, num_chirps=64, num_adcsamples=64, max_num_chirps=1024):
        self.num_chirps = num_chirps
        self.num_adcsamples = num_adcsamples
        self.max_num_chirps = max_num_chirps   
        self.current_matrix = np.empty((0, num_adcsamples), dtype=np.float32)  # 初始化为空矩阵
        self.mutex = QMutex()  # 内置互斥锁

    def append_matrix(self, new_matrix):
        with QMutexLocker(self.mutex):  # 自动加锁
            if new_matrix.shape != (self.num_chirps, self.num_adcsamples):  # 检查新矩阵尺寸
                raise ValueError("New matrix must be of shape (self.num_chirp, self.num_adcsample")

            # 沿第二维拼接新矩阵
            self.current_matrix = np.concatenate((self.current_matrix, new_matrix), axis=0)

            # 如果当前矩阵行数超过最大行数，则移除最早的行
            if self.current_matrix.shape[0] > self.max_num_chirps:
                print("满了")
                # 计算超出的行数
                excess_rows = self.current_matrix.shape[0] - self.max_num_chirps
                # 移除最早的行
                self.current_matrix = self.current_matrix[excess_rows:, :]

    def remove_rows(self, num_rows):
        with QMutexLocker(self.mutex):  # 自动加锁
            # 确保不会尝试移除超过当前矩阵行数的行
            if num_rows > self.current_matrix.shape[0]:
                raise ValueError("Attempting to remove more rows than currently exist.")
            
            # 移除指定数量的列
            self.current_matrix = self.current_matrix[num_rows:, :]

    def get_matrix(self):
        with QMutexLocker(self.mutex):  # 自动加锁
            return self.current_matrix.copy()  # 返回副本以保证数据安全
    
    def get_matrix_rows(self, rows):
        with QMutexLocker(self.mutex):  # 自动加锁
            return self.current_matrix[:rows, :].copy()
    
class UartListener(QThread):
    def __init__(self, name, data_frame_length, com_port):
        QThread.__init__(self)
        self.frame_length = data_frame_length
        if (int(com_port[3:])>=10):
            serial_config.portName = str.encode("\\\\.\\" + com_port,"utf-8")
        else:
            serial_config.portName  = str.encode(com_port,"utf-8")
        dll.openuart(ctypes.byref(serial_config))
        time.sleep(0.1)
        data_to_send = "{\"radar_transmission\":\"enable\"}".encode('utf-8')
        dll.sendData(data_to_send)
        time.sleep(0.5)
    def run(self):
        dll.readData(ctypes.byref(radar_frame))

class Real_time_Data(QThread):
    def __init__(self, name, matrix_fifo: MatrixFIFO):
        QThread.__init__(self)
        self.matrix_fifo = matrix_fifo
    def run(self):  
        lastflag = 0  
        end_time = 0
        while True:  

            time.sleep(0.001) 
            if lastflag != radar_frame.frameNumber:  
                lastflag = radar_frame.frameNumber  
                # start_time = time.time()  # 记录循环开始时间  
                # loop_duration = end_time - start_time  # 计算循环持续时间  
                # end_time = start_time
                
                # print(f"Loop duration: {loop_duration * 1_000_000:.2f} us")  # 转换为微秒并打印 

                # 将 Ctypes 数组转换为 NumPy 数组  
                frame_data_np = np.ctypeslib.as_array(frame_data_array)  
                # int16转成float32 *2/MAX_ADC_VALUE -1
                frame_data_np = frame_data_np.astype(np.float32) * 2 / MAX_ADC_VALUE - 1
                # frame_data_np = generate_radar_data(radar_frame.chirpNumOneFrame, radar_frame.fft1dNum)
                #frame_data_np求和所有维度
                # print(np.sum(frame_data_np))
                
                frame = frame_data_np.reshape((-1, radar_frame.fft1dNum))  
                
                # 将矩阵添加到 FIFO 中  
                self.matrix_fifo.append_matrix(frame)  
                

class DataProcessor(QThread):
    sendLabel = pyqtSignal('PyQt_PyObject')

    def __init__(self, name, indexPosition, matrix_fifo: MatrixFIFO, r3_data):
        QThread.__init__(self)

        self.r3_data = r3_data
        self.num_2dfft = 128
        self.doppler = DopplerAlgo(32, self.num_2dfft, num_ant=1)
        self.matrix_fifo = matrix_fifo
        self.length_max = 1024
        self.udoppler_queue = deque(maxlen=self.length_max)
        self.udoppler_queue.clear()  # 清空 deque  
        self.count = 0 #用于计算保存多少帧的
        self.ctc_weight = 0.5
        self.reverse_weight = 0.0
        self.mode = 'CPBS_with_left_hyps+atention_rescoring'
        self.model, self.model_streaming = self.load_model()
        self.udoppler_queue_recognize = deque(maxlen=self.model_streaming.decoding_window)
        self.label = "start"
        self.folder_name = 'datas4'
        # 检查文件夹是否存在
        if not os.path.exists(self.folder_name):
            # 如果文件夹不存在，则创建它
            os.makedirs(self.folder_name)
        # 读取文件夹中所有的 .npy 文件
        # self.file_number = len([f for f in os.listdir(self.folder_name) if f.endswith('.npy')])
        # 打开 txt 文件用于记录
        self.log_file = open(f"{self.folder_name}/file_log.txt", 'a')

    def load_model(self):
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[  
                    logging.FileHandler('output.log', encoding='utf-8'),  # 指定日志输出文件名  
                    logging.StreamHandler()  # 可选：同时输出到控制台  
                ] 
        )
        model_path = 'model/quantized_model/final.zip'  # 或者 'wenet/model_average/quantized_model/final_quantized.zip'  
        model = torch.jit.load(model_path) 
        model.eval()

        subsampling_rate = 4
        right_context = 6
        model_streaming = Streaming_ASRModel(model, subsampling_rate, right_context , 
                                             gram_file_txt = "model/2_garm.txt", mode = 'CPBS_with_left_hyps+atention_rescoring',
                                             decoding_chunk_size = 10)
                                
        return model,  model_streaming

    def setlabelinit(self):
        self.label = "start"
        self.sendinfo(self.label,"blue")
    
    def sendinfo(self, string, fontcolor):
        d = dict()
        d["string"] = "请写入:"+string
        d["fontcolor"] = fontcolor
        self.sendLabel.emit(d)

    
    def run(self):
        count = 0
        count1 = 0
        conn = 0
        index = 0
        isStarted = True
        isRecongnize = False
        hyp_list = ()
        self.setPriority(QThread.HighPriority) 
        # print("开始")
        logging.info("The next recognition target is:{}".format(self.label))
        while(True):
            sizematrix = self.matrix_fifo.get_matrix().shape
            # print(sizematrix[0]) #打印出来，感觉没有晒满导致丢数据
            if sizematrix[0]>=self.num_2dfft:
                
                frame = self.matrix_fifo.get_matrix_rows(self.num_2dfft)
                # 由于这里的导致数据不是实时的
                self.matrix_fifo.remove_rows(16)
                # print(np.sum(frame))
                rd_spectrum = self.doppler.compute_doppler_map(frame, 0)
                udoppler1D = np.sum(np.abs(rd_spectrum),axis=0, dtype=np.float32)
                # print(np.sum(udoppler1D))
                self.udoppler_queue.append(udoppler1D)
                self.udoppler_queue_recognize.append(udoppler1D)
                count1 += 1
                pwr = np.sum(np.array(self.udoppler_queue_recognize)[-10:], axis=-1)
                # 获取最新的10个udoppler1D数组（形状为 [10, 128]）
                udopplers = np.array(self.udoppler_queue_recognize)[-10:]

                # 定义要排除的中心5个点的索引（假设128维数组）
                exclude_start = 62  # 中心点索引范围：62, 63, 64, 65, 66
                exclude_end = 67    # 67是切片终止索引（不包含）

                # 创建保留区域的索引（合并前半段和后半段）
                valid_indices = np.r_[0:exclude_start, exclude_end:128]

                # 排除中心5个点后的数组（形状变为 [10, 123]）
                filtered_udopplers = udopplers[:, valid_indices]

                # 对每个处理后的数组求和（沿最后一个轴，得到10个和值）
                pwr = np.sum(filtered_udopplers, axis=-1)
                # print(pwr)
                with torch.no_grad():
                    if len(self.udoppler_queue) > 256 and np.all(pwr>0.046):
                        if(isStarted):
                            logging.info("Starting recognition")
                            outputs = []
                            hyp_list = ()
                            count= 0
                            conn = 0
                            index = 0
                            offset = 0
                            processing_time_sum= 0
                            subsampling_cache: Optional[torch.Tensor] = None
                            elayers_output_cache: Optional[List[torch.Tensor]] = None
                            conformer_cnn_cache: Optional[List[torch.Tensor]] = None
                            content = ''
                            isStarted = False
                            isRecongnize = True
                            
                    if(isStarted== False and isRecongnize):  
                        if conn>0:
                            conn = conn -1
                            if conn == 0:
                                isStarted = True
                                isRecongnize = False

                    # print(len(self.udoppler_queue_recognize))
                        pwr1 = np.sum(np.array(self.udoppler_queue_recognize)[-10:], axis=-1)
                        
                        if(np.all(pwr1[:-1]>0.05) and pwr1[-1]<0.05):
                            # print("结束识别pwr1= %f\n" % pwr1[-1])
                            conn = 50

                        if(1):
                            count  += 1
                            
                            if (count % self.model_streaming.stride==0):
                                index += 1
                                # 开始计时
                                process_start = time.perf_counter()

                                chunk_xs = np.abs(np.array(self.udoppler_queue_recognize))
                                chunk_xs = np.expand_dims(chunk_xs, axis=0)
                                chunk_xs = torch.Tensor(chunk_xs)
                                (encoder_out, subsampling_cache, elayers_output_cache,
                                conformer_cnn_cache) = self.model.encoder.forward_chunk(
                                                        chunk_xs, 
                                                        offset, 
                                                        self.model_streaming.required_cache_size,
                                                        subsampling_cache,
                                                        elayers_output_cache,
                                                        conformer_cnn_cache
                                                        )
                                outputs.append(encoder_out)
                                offset += encoder_out.size(1)
                                ys = torch.cat(outputs, 1)
                                
                                if self.mode == 'CPBS_with_left_hyps':
                                    hyp_list, _ = self.model_streaming.steaming_CPBS_with_left_hyps(ys, index, hyp_list)
                                elif self.mode  == 'CPBS_without_left_hyps':
                                    hyp_list = self.model_streaming.steaming_CPBS_without_left_hyps(encoder_out, index, hyp_list)
                                elif self.mode  == 'CGS_with_left_hyps':
                                    hyp_list = self.model_streaming.steaming_CGS_with_left_hyps(ys, index, hyp_list)
                                elif self.mode  == 'CGS_without_left_hyps':  
                                    hyp_list = self.model_streaming.steaming_CGS_without_left_hyps(encoder_out, index, hyp_list)
                                elif self.mode  == 'CPBS_with_left_hyps+atention_rescoring':
                                    hyp_list, hyps = self.model_streaming.steaming_CPBS_with_left_hyps(ys, index, hyp_list)
                                processing_time = time.perf_counter() - process_start
                                processing_time_sum = processing_time_sum + processing_time
                                if index ==1:
                                    logging.info("First token latency (ms): {:.8f}".format(processing_time*1000.0))

                    if(len(hyp_list)>0 and isRecongnize==False):
                        
                        if self.mode == 'CPBS_without_left_hyps':
                            # 有优点也有缺点，无法处理连符号。在cps已经把blank去除了。gs算法不影响
                            hyp_list = [remove_duplicates_and_blank(hyp) for hyp in [hyp_list]]
                            hyp_list = hyp_list[0]

                        for w in hyp_list:
                            content += id_to_char[w]
                        logging.info("CTC Decoder's string:{}".format(content))

                        if self.mode == 'CPBS_with_left_hyps+atention_rescoring' :
                            attention_rescoring_start = time.perf_counter()
                            hyp, _ = self.model_streaming.attention_rescoring_chunk(ys, hyps, ctc_weight=self.ctc_weight,
                                            reverse_weight=self.reverse_weight)
                            attention_rescoring_time = time.perf_counter() - attention_rescoring_start
                            logging.info("Attention rescoring latency (ms): {:.8f}".format(attention_rescoring_time*1000.0))
                            Recogntion_time = processing_time_sum + attention_rescoring_time
                            rtf = Recogntion_time / (index *self.model_streaming.decoding_window*16.0/64.0*0.0384)
                                
                            logging.info("RTF: {:.8f}".format(rtf))
                            content = ''
                            for w in hyp:
                                content += id_to_char[w]   
                            if content == self.label:
                                self.sendinfo(content,"green")
                            else:
                                self.sendinfo(content,"red")
                            logging.info("The string after attention restoring:{}".format(content))   
                        hyp_list = ()
                        logging.info("Recognition ended\n")

                        # 保存数据

                        length = min(count, self.length_max)
                        saveudoppler = np.array(self.udoppler_queue)[-length:-1,:]
                        self.range_queue = np.ones((length + 1, 32))    # 第二维度为32  
                        self.xzflag_queue = np.ones((length + 1, 3), dtype=np.float32)  #
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
                        

                        self.label = common_words[np.random.randint(0, len(common_words))]
                        logging.info("The next recognition target is:{}".format(self.label))
                        self.sendinfo(self.label,"blue")

                self.r3_data.put(np.array(self.udoppler_queue))
                
