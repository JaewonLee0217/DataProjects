import torch
import torch.nn as nn
import torchvision.models as models

# LSTM 기반의 Caption 생성
# 추출한 특징을 입력으로 받아 이미지에 대한 캡션을 생성.

class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def forward(self, features, captions):
        pass