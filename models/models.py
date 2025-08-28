from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super().__init__()
        # EfficientNet-b0 백본
        self.backbone = EfficientNet.from_pretrained(model_name)
        # 백본 파라미터 동결
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # skip connection을 위한 feature map 저장
        self.feature_maps = []

        # Forward Hook 등록할 인덱스
        self.hook_ids = [2, 4, 8, 15]
        # Forward Hook 함수 정의
        def save_hook(module, input, output):
            self.feature_maps.append(output)
        # conv_stem과 지정된 블록들에 Hook 등록
        self.backbone._conv_stem.register_forward_hook(save_hook)
        for idx in self.hook_ids:
            self.backbone._blocks[idx].register_forward_hook(save_hook)
        
    def forward(self, x):
        self.feature_maps = []  # 초기화
        x = self.backbone.extract_features(x)  # EfficientNet의 feature extractor
        return x, self.feature_maps

# Skip connection을 포함한 디코더 블록
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 업샘플링
        self.up_conv = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2,stride=2)
        # 더블 컨볼루션 블록
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1) # 스킵 연결
        x = self.convblock(x)
        return x

class EfficientUnet(nn.Module):
    def __init__(self, num_classes=1, model_name='efficientnet-b0'):
        super().__init__()
        self.encoder = EfficientNetEncoder(model_name=model_name)
        self.decoder4 = DecoderBlock(320, 112, 256)
        self.decoder3 = DecoderBlock(256, 40, 128)
        self.decoder2 = DecoderBlock(128, 24, 64)
        self.decoder1 = DecoderBlock(64, 32, 32)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32,16,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,num_classes,kernel_size=1)
        )
    
    def forward(self, x):
        x, features = self.encoder(x) # 인코더의 스킵 연결 특징 맵 추출
        x = self.decoder4(features[-1], features[-2]) # 스킵 연결 적용하여 디코더 블록 통과
        x = self.decoder3(x, features[-3])
        x = self.decoder2(x, features[-4])
        x = self.decoder1(x, features[0])
        x = self.final_conv(x) # 최종 출력 레이어
        return x