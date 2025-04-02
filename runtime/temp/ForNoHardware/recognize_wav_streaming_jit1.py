from typing import Optional, List, Tuple
import logging

import torch

import numpy as np


from recognize_streaming import remove_duplicates_and_blank
from recognize_streaming import char_to_id, id_to_char, Streaming_ASRModel


def main():

    mode  = 'CPBS_with_left_hyps+atention_rescoring'
    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[  
            logging.FileHandler('output.log', encoding='utf-8'),  # 指定日志输出文件名  
            logging.StreamHandler()  # 可选：同时输出到控制台  
        ] 
    )
    
    model_path = 'runtime/temp/ForDemoBGT60TR13Kit/model/quantized_model/final.zip'  # 或者 'wenet/model_average/quantized_model/final_quantized.zip'  
    model = torch.jit.load(model_path) 
    model.eval()
    subsampling_rate = 4
    right_context = 6
    model_streaming = Streaming_ASRModel(model, subsampling_rate, right_context , 
                                            gram_file_txt = "runtime/temp/ForDemoBGT60TR13Kit/model/2_garm.txt", mode = mode,
                                            decoding_chunk_size = 10)


    with torch.no_grad():
        matrix = np.load("dataset/example/datas2/data_20240505_171357.npy")
        # Slice the matrix into three parts
        matrix_128 = matrix[:, :128]    # First 128 columns
        matrix_128 = np.expand_dims(matrix_128, axis=0)  # Add dimension at first position
        feats = torch.from_numpy(matrix_128).float() 

        target = np.expand_dims("miss", axis=0) 
        feats_lengths = matrix_128.shape[1]

        outputs = []
        hyp_list = []
        offset = 0

        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        content = ''

        # Feed forward overlap input step by step
        for cur in range(0, feats_lengths - model_streaming.context + 1, model_streaming.stride):
            end = min(cur + model_streaming.decoding_window, feats_lengths) 
            chunk_xs = feats[:, cur:end, :] # (1, max_len, dim) bacth_size必须设置为1

            (encoder_out, subsampling_cache, elayers_output_cache,
            conformer_cnn_cache) = model.encoder.forward_chunk(
                                    chunk_xs, 
                                    offset, 
                                    model_streaming.required_cache_size,
                                    subsampling_cache,
                                    elayers_output_cache,
                                    conformer_cnn_cache
                                    )
            outputs.append(encoder_out)
            offset += encoder_out.size(1)
            ys = torch.cat(outputs, 1)
            index = cur//model_streaming.stride
            if mode == 'CPBS_with_left_hyps':
                hyp_list, _ = model_streaming.steaming_CPBS_with_left_hyps(ys, index, hyp_list)
            elif mode == 'CPBS_without_left_hyps':
                hyp_list = model_streaming.steaming_CPBS_without_left_hyps(encoder_out, index, hyp_list)
            elif mode == 'CGS_with_left_hyps':
                hyp_list = model_streaming.steaming_CGS_with_left_hyps(ys, index, hyp_list)
            elif mode == 'CGS_without_left_hyps':  
                hyp_list = model_streaming.steaming_CGS_without_left_hyps(encoder_out, index, hyp_list)
            elif mode == 'CPBS_with_left_hyps+atention_rescoring':
                hyp_list, hyps = model_streaming.steaming_CPBS_with_left_hyps(ys, index, hyp_list)

        if mode == 'CPBS_without_left_hyps':
            # 有优点也有缺点，无法处理连符号。在cps已经把blank去除了。gs算法不影响
            hyp_list = [remove_duplicates_and_blank(hyp) for hyp in [hyp_list]]
            hyp_list = hyp_list[0]

        for w in hyp_list:
            content += id_to_char[w]
        logging.info("chunk产生的字符串为:{}".format(content))

        if mode == 'CPBS_with_left_hyps+atention_rescoring' :
            hyp, _ = model_streaming.attention_rescoring_chunk(ys, hyps, ctc_weight=0.5,
                            reverse_weight=0.0)
            content = ''
            for w in hyp:
                content += id_to_char[w]   
            logging.info("重新打分后产生的字符串为:{}".format(content))   

        logging.info("正确字符串为:{}".format(target[0]))



if __name__ == '__main__':
    main()


