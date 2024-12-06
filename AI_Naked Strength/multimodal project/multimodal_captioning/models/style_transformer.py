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
    # 설명 적용 필요
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 짝수 인덱스에는 싸인, 홀수 인덱스에는 코사인 적용 차별성 유지, 상대적 거리 계산 효율성
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term) #

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StyleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout=0.1):
        super(StyleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)