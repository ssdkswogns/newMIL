import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from models.inceptiontime import InceptionTimeFeatureExtractor
from models.nystrom_attention import NystromAttention



# from inceptiontime import InceptionTimeFeatureExtractor
# from nystrom_attention import NystromAttention
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + attn_out
            return x, attn
        x = x + self.attn(self.norm(x))
        return x

    

### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift): #size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
  
    x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1]).cuda()
    # print(x.shape)
    x = x.reshape(1,-1).repeat(size[0],1)
    # print(x.shape)
    # print(shift.shape)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / ( 3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

    return wavelet #d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len=256, hidden_len=512, dropout=0.0):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3, num_cls_tokens=1):
        # x: (B, num_cls_tokens + N, D)
        cls_tokens = x[:, :num_cls_tokens]     # (B, C, D)
        feat_token = x[:, num_cls_tokens:]     # (B, N, D)

        x_feat = feat_token.transpose(1, 2)    # (B, D, N)

        D = x_feat.shape[1]
        scale1, shift1 = wave1[0, :], wave1[1, :]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D, 19), scale=scale1, shift=shift1)
        scale2, shift2 = wave2[0, :], wave2[1, :]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D, 19), scale=scale2, shift=shift2)
        scale3, shift3 = wave3[0, :], wave3[1, :]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D, 19), scale=scale3, shift=shift3)

        pos1 = F.conv1d(x_feat, wavelet_kernel1.unsqueeze(1), groups=D, padding='same')
        pos2 = F.conv1d(x_feat, wavelet_kernel2.unsqueeze(1), groups=D, padding='same')
        pos3 = F.conv1d(x_feat, wavelet_kernel3.unsqueeze(1), groups=D, padding='same')

        feat_token = feat_token + self.proj_1(pos1.transpose(1, 2) + pos2.transpose(1, 2) + pos3.transpose(1, 2))

        x = torch.cat((cls_tokens, feat_token), dim=1)   # (B, C+N, D)
        return x
    
def _crop_nystrom_attn(attn, n_tokens, num_landmarks):
    """
    attn: (B, H, n_pad, n_pad) or (B, n_pad, n_pad)
    n_tokens: padding 전 토큰 길이 (CLS 포함)
    num_landmarks: NystromAttention.num_landmarks (m)
    """
    m = num_landmarks
    p = (m - (n_tokens % m)) % m  # left padding length

    if attn is None:
        return None

    if attn.dim() == 4:
        return attn[..., p:p + n_tokens, p:p + n_tokens]
    elif attn.dim() == 3:
        return attn[:, p:p + n_tokens, p:p + n_tokens]
    else:
        raise ValueError(f"Unexpected attn shape: {attn.shape}")


class TimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0., is_instance=False):
        super().__init__()
        self.n_classes = n_classes
        self.is_instance = is_instance
     

        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
       

            
        # define WPE    
        self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim))
        self.wave1 = torch.randn(2, mDim,1 )
        self.wave1[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim,1 )
        self.wave2[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim,1 )
        self.wave3[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim,1 )
        self.wave1_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave1_ = nn.Parameter(self.wave1)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3)        
            
            
            
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        # self._fc2 = nn.Sequential(
        #     nn.Linear(mDim,mDim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(mDim, n_classes)
        # )
        
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        

        
        initialize_weights(self)
        
        
    def forward(self, x, warmup=False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x = x1

        B, seq_len, D = x.shape
        global_token = x.mean(dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # WPE1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3, num_cls_tokens=self.n_classes)

        if self.is_instance:
            x, attn_layer1 = self.layer1(x, return_attn=True)
        else:
            x = self.layer1(x)

        # WPE2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_, num_cls_tokens=self.n_classes)

        if self.is_instance:
            x, attn_layer2 = self.layer2(x, return_attn=True)
        else:
            x = self.layer2(x)

        # cls token만 사용
        x_cls = x[:, :self.n_classes, :]

        if warmup:
            x_cls = 0.1 * x_cls + 0.99 * global_token[:, None, :]

        # logits = self._fc2(x_cls)
        logits = x_cls.mean(dim=2)

        if self.is_instance:
            # 여기서 n_tokens는 padding 전 토큰 길이 (CLS 포함)
            n_tokens = self.n_classes + seq_len
            m1 = self.layer1.attn.num_landmarks
            m2 = self.layer2.attn.num_landmarks

            attn_layer1 = _crop_nystrom_attn(attn_layer1, n_tokens, m1)
            attn_layer2 = _crop_nystrom_attn(attn_layer2, n_tokens, m2)

            return logits, x_cls, attn_layer1, attn_layer2

        return logits
    
class originalTimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0., is_instance=False):
        super().__init__()
        self.n_classes = n_classes
        self.is_instance = is_instance
     

        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
       

            
        # define WPE    
        # self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim,1 )
        self.wave1[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim,1 )
        self.wave2[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim,1 )
        self.wave3[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim,1 )
        self.wave1_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave1_ = nn.Parameter(self.wave1)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3)        
            
            
            
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )
        
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        

        
        initialize_weights(self)
        
        
    def forward(self, x, warmup=False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x = x1

        B, seq_len, D = x.shape
        global_token = x.mean(dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # WPE1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3, num_cls_tokens=1)

        if self.is_instance:
            x, attn_layer1 = self.layer1(x, return_attn=True)
        else:
            x = self.layer1(x)

        # WPE2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_, num_cls_tokens=1)

        if self.is_instance:
            x, attn_layer2 = self.layer2(x, return_attn=True)
        else:
            x = self.layer2(x)

        # cls token만 사용
        # x_cls = x[:, :self.n_classes, :]
        x_cls = x[:, 0]

        if warmup:
            x_cls = 0.1 * x_cls + 0.99 * global_token

        logits = self._fc2(x_cls)
        # logits = x_cls.mean(dim=2)

        if self.is_instance:
            # 여기서 n_tokens는 padding 전 토큰 길이 (CLS 포함)
            n_tokens = self.n_classes + seq_len
            m1 = self.layer1.attn.num_landmarks
            m2 = self.layer2.attn.num_landmarks

            attn_layer1 = _crop_nystrom_attn(attn_layer1, n_tokens, m1)
            attn_layer2 = _crop_nystrom_attn(attn_layer2, n_tokens, m2)

            return logits, x_cls, attn_layer1, attn_layer2

        return logits

if __name__ == "__main__":
    x = torch.randn(3, 400, 4).cuda()
    model = TimeMIL(in_features=4,mDim=128).cuda()
    ylogits =model(x)
    print(ylogits.shape)
