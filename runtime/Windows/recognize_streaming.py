# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import math
from typing import Optional, List, Tuple
import logging
from torch.nn.utils.rnn import pad_sequence
import torch

char_to_id = {  '<blank>': 0,
                '<unk>': 1,
                ' ': 2,
                'a': 3,
                'b': 4,
                'c': 5,
                'd': 6,
                'e': 7,
                'f': 8,
                'g': 9,
                'h': 10,
                'i': 11,
                'j': 12,
                'k': 13,
                'l': 14,
                'm': 15,
                'n': 16,
                'o': 17,
                'p': 18,
                'q': 19,
                'r': 20,
                's': 21,
                't': 22,
                'u': 23,
                'v': 24,
                'w': 25,
                'x': 26,
                'y': 27,
                'z': 28,
                'I': 29,
                '<sos/eos>': 30
            }

id_to_char = {value: key for key, value in char_to_id.items()}

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp

def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad

def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad

def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(lengths.size(0))
    max_len = int(lengths.max().item())
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

class Streaming_ASRModel():
    def __init__(
        self,
        model,
        subsampling_rate: int = 4,
        right_context: int = 6,
        beam_size: int =5, 
        gram_file_txt: str = ' ',
        LM_weight: float = 10,
        mode: str = '',
        decoding_chunk_size: int = 10,
        num_decoding_left_chunks: int = -1,
        ):
        super().__init__()
        self.model = model
        self.sequence_probs = self.get_gram_prod(gram_file_txt)
        self.LM_weight = LM_weight
        self.beam_size = beam_size
        self.mode = mode

        subsampling =  subsampling_rate
        self.context = right_context + 1  # Add current frame
        self.stride = subsampling * decoding_chunk_size
        self.decoding_window = (decoding_chunk_size - 1) * subsampling + self.context

        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks #-1则是取历史所有数据

    def get_gram_prod(self, filepath):
        sequence_probs = {}
        with open(filepath) as f: data = f.read()
        for line in data.strip().split('\n'):
            parts = line.split()  # 分割每一行
            if len(parts) >= 2:
                key = parts[0]      # 第一列是键
                value = float(parts[1])  # 第二列转换为浮点数
                sequence_probs[key] = value  # 存储到字典
        return sequence_probs
    
    def decoder_chunk_with_mode(self, encoder_out, index, hyp_list, mode = None):
        if mode is not None:
            mode = mode
        else:
            mode = self.mode
        if mode == 'CPBS_with_left_hyps':
            hyp_list = self.steaming_CPBS_with_left_hyps(encoder_out, index, hyp_list)
            return hyp_list
        elif mode == 'CPBS_without_left_hyps':
            hyp_list = self.steaming_CPBS_without_left_hyps(encoder_out, index, hyp_list)
            return hyp_list
        elif mode == 'CGS_with_left_hyps':
            hyp_list = self.steaming_CGS_with_left_hyps(encoder_out, index, hyp_list)
            return hyp_list
        elif mode == 'CGS_without_left_hyps':  
            hyp_list = self.steaming_CGS_without_left_hyps(encoder_out, index, hyp_list)
            return hyp_list


    def steaming_ctc_prefix_beam_search_with_LM(
        self,
        encoder_out: torch.Tensor,
        beam_size: int =5 ,
        sequence_probs: dict = None,
        LM_weight: float = 0,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            encoder_out (torch.Tensor): (batch, max_len, feat_dim)
            encoder_mask (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        maxlen = encoder_out.size(1)
        ctc_probs = self.model.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)时间复杂度为O(t*m*m), 最大可以设置为vocab_size  时间复杂度为O(t*m*N)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    # Adding language model probability
                    if self.sequence_probs is not None:
                        lm_prob = self.lm_score(self.sequence_probs, self.LM_weight, prefix, s)
                    else:
                        lm_prob = 0

                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps + lm_prob, pnb + ps + lm_prob])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps + lm_prob])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix] #n_pnb没有用新的
                        n_pnb = log_add([n_pnb, pb + ps + lm_prob])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps + lm_prob, pnb + ps + lm_prob])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                                key=lambda x: log_add(list(x[1])),
                                reverse=True)
            cur_hyps = next_hyps[:self.beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out
    def lm_score(self, sequence_probs, LM_weight, prefix, s):
        last = prefix[-1] if len(prefix) > 0 else 0
        key = id_to_char[last]+id_to_char[s]
        # print(key)
        if key in sequence_probs:
            value = sequence_probs[key] * LM_weight #10是权重
            # print(value)
        else:
            # print(key) 应该给0,而不是给无穷，虽然直觉上应该是无穷，这里给0上面的应该给正数
            value = 0
            # value = -float('inf')
            # value = 0
        return value
    def steaming_CPBS_with_left_hyps(self, ys, index, hyp_list):
        # 这个是所有进行重新搜索
        hyps, _  = self.steaming_ctc_prefix_beam_search_with_LM(ys)
        
        hyp, _ = hyps[0]
        temp_content = ''
        for w in hyp:
            temp_content += id_to_char[w]
        hyp_list = hyp
        logging.info("The generated string for chunk 0~{} is:{}".format(index, temp_content))
        return hyp_list, hyps

    def steaming_CPBS_without_left_hyps(self, encoder_out, index, hyp_list):
        hyps, _  = self.steaming_ctc_prefix_beam_search_with_LM(
                            encoder_out)

        hyp, _ = hyps[0]
        temp_content = ''

        for w in hyp:
            temp_content += id_to_char[w]
            hyp_list.append(w)
        

        logging.info("第{}个chunk产生的字符串为:{}".format(index, temp_content))
        return hyp_list
    
    def steaming_CGS_with_left_hyps(self, encoder_out, index, hyp_list):
        chunk_length = encoder_out.size(1)
        batch_size = encoder_out.size(0)
        encoder_out_lens = torch.tensor([chunk_length])

        # 将张量移动到CUDA设备
        encoder_out_lens = encoder_out_lens.to(encoder_out.device)

        ctc_probs = self.model.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, chunk_length)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.model.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        hyp = hyps[0]
        temp_content = ''

        for w in hyp:
            temp_content += id_to_char[w]

        hyp_list = hyp
        logging.info("第{}个chunk产生的字符串为:{}".format(index, temp_content))

        return hyp_list

    def steaming_CGS_without_left_hyps(self, encoder_out, index, hyp_list):
        chunk_length = encoder_out.size(1)
        batch_size = encoder_out.size(0)
        encoder_out_lens = torch.tensor([chunk_length])

        # 将张量移动到CUDA设备
        encoder_out_lens = encoder_out_lens.to(encoder_out.device)
        ctc_probs = self.model.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, chunk_length)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.model.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        hyp = hyps[0]
        temp_content = ''

        for w in hyp:
            temp_content += id_to_char[w]
            hyp_list.append(w)


        logging.info("第{}个chunk产生的字符串为:{}".format(index, temp_content))

        return hyp_list
    
    def attention_rescoring_chunk(
        self,
        encoder_out: torch.Tensor,
        hyps: list,
        ctc_weight: float = 0.0,
        reverse_weight: float = 0.0,
    ):

        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.model.decoder, 'right_decoder')

        device = encoder_out.device

        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.model.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.model.sos, self.model.eos, self.model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(self.beam_size, 1, 1)
        encoder_mask = torch.ones(self.beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.model.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.model.sos, self.model.eos,
                                    self.model.ignore_id)
        decoder_out, r_decoder_out, _ = self.model.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.model.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score
    

