import numpy as np
import os
# import threading as th
from collections import deque
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from helpers.DopplerAlgo import *
from typing import Optional, List, Tuple
import logging
import os
import sys
import torch
import yaml

from PyQt5.QtCore import QThread, pyqtSignal
from recognize_streaming import Streaming_ASRModel, remove_duplicates_and_blank, id_to_char


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
            print("满了")
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
        with DeviceFmcw() as device:

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
        self.label = ''
        self.sendinfo(self.label,"blue")
    
    def sendinfo(self, string, fontcolor):
        d = dict()
        d["string"] = "请写入:"+string
        d["fontcolor"] = fontcolor
        self.sendLabel.emit(d)

    def run(self):
        count = 0
        conn = 0
        index = 0
        isStarted = True
        isRecongnize = False
        hyp_list = ()
        # print("开始")
        while(True):
            sizematrix = self.matrix_fifo.get_matrix().shape
            # print(sizematrix[0]) #打印出来，感觉没有晒满导致丢数据
            if sizematrix[0]>=self.num_2dfft:
                
                frame = self.matrix_fifo.get_matrix_rows(self.num_2dfft)
                
                # 由于这里的导致数据不是实时的
                self.matrix_fifo.remove_rows(16)
                rd_spectrum = self.doppler.compute_doppler_map(frame, 0)
                udoppler1D = np.sum(np.abs(rd_spectrum),axis=0, dtype=np.float32)
                self.udoppler_queue.append(udoppler1D)
                self.udoppler_queue_recognize.append(udoppler1D)
                pwr = np.sum(np.array(self.udoppler_queue_recognize)[-10:], axis=-1)
                # print("pwr= %f, pwr1= %f\n, len = %d" % (pwr,pwr1, len(self.udoppler_queue)))

                # print("pwr= %f\n" % pwr)
                # print(self.xzflag_queue)
                
                with torch.no_grad():
                    if len(self.udoppler_queue) > 256 and np.all(pwr>0.1):
                        if(isStarted):
                            logging.info("Starting recognition")
                            outputs = []
                            hyp_list = ()
                            count= 0
                            conn = 0
                            index = 0
                            offset = 0
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
                                chunk_xs = np.array(self.udoppler_queue_recognize)
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

                    if(len(hyp_list)>0 and isRecongnize==False):
                        
                        if self.mode == 'CPBS_without_left_hyps':
                            # 有优点也有缺点，无法处理连符号。在cps已经把blank去除了。gs算法不影响
                            hyp_list = [remove_duplicates_and_blank(hyp) for hyp in [hyp_list]]
                            hyp_list = hyp_list[0]

                        for w in hyp_list:
                            content += id_to_char[w]
                        logging.info("chunk产生的字符串为:{}".format(content))

                        if self.mode == 'CPBS_with_left_hyps+atention_rescoring' :
                            hyp, _ = self.model_streaming.attention_rescoring_chunk(ys, hyps, ctc_weight=self.ctc_weight,
                                            reverse_weight=self.reverse_weight)
                            content = ''
                            for w in hyp:
                                content += id_to_char[w]   
                            logging.info("重新打分后产生的字符串为:{}".format(content))   
                        hyp_list = ()
                        print("结束识别")

                # self.r3_data.put(np.array(self.udoppler_queue))
                

