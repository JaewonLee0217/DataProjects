import torch
import torch.nn as nn
import math
'''
1. PositionalEncoding 클래스: 
    Transformer 에서 순서 정보를 인코딩하는 데 사용
    
2. StyleTransformer 클래스:
    __init__ 메서드: 임베딩 층, 위치 인코딩, Transformer 모델, 출력 층을 초기화
    forward 메서드: 입력 시퀀스를 변환하여 스타일이 변경된 출력을 생성
    generate 메서드: 주어진 입력에 대해 새로운 시퀀스를 생성

'''



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()


class StyleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StyleTransformer, self).__init__()