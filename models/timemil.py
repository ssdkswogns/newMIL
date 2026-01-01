import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from models.inceptiontime import InceptionTimeFeatureExtractor
from models.nystrom_attention import NystromAttention, Attention, SDPAttention



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
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2,dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        out, attn = self.attn(x=self.norm(x),return_attn=True)
        x = x + out
        return x, attn
    
class TransLayer_new(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim은 num_heads로 나누어 떨어져야 합니다."
        self.norm = norm_layer(dim)
        self.attn = Attention(                  # <= 질문에 주신 심플 Attention
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )

    def forward(self, x):
        # Pre-Norm
        x_norm = self.norm(x)
        out, attn = self.attn(x_norm, return_attn=True)
        x = x + out
        return x, attn
    
class TransLayer_amb(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim은 num_heads로 나누어 떨어져야 합니다."
        self.norm = norm_layer(dim)
        self.attn = SDPAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )

    def forward(self, x):
        # Pre-Norm
        x_norm = self.norm(x)
        out= self.attn(x_norm)
        x = x + out
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
    def __init__(self, dim=512, max_len = 256,hidden_len = 512,dropout=0.0):
        super().__init__()
       
        #n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)
        
         
        
    def forward(self,x,wave1,wave2,wave3,n_cls=1):
        cls_token, feat_token = x[:, :n_cls], x[:, n_cls:]
        
        x = feat_token.transpose(1, 2)
        
        
        D = x.shape[1]
        scale1, shift1 =wave1[0,:],wave1[1,:]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D,19), scale=scale1, shift=shift1)
        scale2, shift2 =wave2[0,:],wave2[1,:]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D,19), scale=scale2, shift=shift2)
        scale3, shift3 =wave3[0,:],wave3[1,:]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D,19), scale=scale3, shift=shift3)
        
         #Eq. 11
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2)   #B*N*D
        # print(x.shape)
        
        

        #Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2)+pos2.transpose(1, 2)+pos3.transpose(1, 2))# + mixup_encording
        
        # mixup token information
        
        x = torch.cat((cls_token, x), dim=1)
        return x


class TimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0., is_instance=False):
        super().__init__()
        self.is_instance = is_instance

        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
       

            
        # define WPE    
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
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3_)        
            
            
            
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
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
        
        
    def forward(self, x,warmup=False):
        

        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x= x1

        

        B, seq_len, D = x.shape

        
        view_x = x.clone()
        
      
        global_token = x.mean(dim=1)#[0]
      
        
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
      
        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3)
      
        # TransLayer x1
        x, attn_layer1 = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x,self.wave1_,self.wave2_,self.wave3_)
        
       
        # TransLayer x2
        
       
        x, attn_layer2 = self.layer2(x)
       
        # only cls_token is used for cls
        x = x[:,0]
   
        
        # stablity of training random initialized global token
        if warmup:
            x = 0.1*x+0.99*global_token   
 
        logits = self._fc2(x)
            
        if self.is_instance:
            return logits, x, attn_layer1, attn_layer2
        else:
            return logits
    

class newTimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0., is_instance=False):
        super().__init__()
        self.is_instance = is_instance
     
        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
        # define WPE    
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
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3_)        
            
            
            
            
        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim)) # (1,C,D)
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer_new(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer_new(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        ) 
        
        
        self.alpha = nn.Parameter(torch.ones(1))
        self.n_classes = n_classes
        
        initialize_weights(self)
        
        
    def forward(self, x,warmup=False):
        

        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x= x1

        

        B, seq_len, D = x.shape
        C = self.n_classes        
      
        global_token = x.mean(dim=1)#[0]
      
        
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
      
        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3,self.n_classes)
        # TransLayer x1
        x, attn_layer1 = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x,self.wave1_,self.wave2_,self.wave3_,self.n_classes)
        # TransLayer x2
        x, attn_layer2 = self.layer2(x)
       
        # only cls_token is used for cls (assume last cls embedding as background)
        x = x[:,:C,:]
        
        # # stablity of training random initialized global token
        # if warmup:
        #     x = 0.1*x+0.99*global_token.unsqueeze(1)   
 
        # logits = self._fc2(x)
        logits = x.mean(dim=2)

        if self.is_instance:
            return logits, x, attn_layer1, attn_layer2
        else:
            return logits

class AmbiguousMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0., is_instance=False):
        super().__init__()
        self.is_instance = is_instance
     
        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        
        # define WPE    
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
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim,1 )
        self.wave2_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim,1 )
        self.wave3_[0]=torch.ones( mDim,1 )+ torch.randn( mDim,1 ) #make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3_)        

        hidden_len = 2* max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim)) # (1,C,D)
        self.pos_layer =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        self.pos_layer2 =  WaveletEncoding(mDim,max_seq_len,hidden_len) 
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer_new(dim=mDim,dropout=dropout)
        self.layer2 = TransLayer_new(dim=mDim,dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )
        
        self.cls_reduce = nn.Conv1d(in_channels=mDim, out_channels=1, kernel_size=1, bias=True)
        # self.instance_head = nn.Linear(mDim, n_classes)
        self.instance_head = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.n_classes = n_classes
        
        initialize_weights(self)
        
        
    def forward(self, x,warmup=False):

        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x= x1

        B, seq_len, D = x.shape
        C = self.n_classes        

        cls_tokens = self.cls_token.expand(B, -1, -1)    #B * 1 * d
      
        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x,self.wave1,self.wave2,self.wave3,self.n_classes)
        # TransLayer x1
        x, attn_layer1 = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x,self.wave1_,self.wave2_,self.wave3_,self.n_classes)
        # TransLayer x2
        x, attn_layer2 = self.layer2(x)

        x_cls = x[:,:C,:]
        x_cls_chfirst = x_cls.permute(0, 2, 1)

        x_seq = x[:, C:, :]
        instance_logits = self.instance_head(x_seq)
        
        logits = self.cls_reduce(x_cls_chfirst).squeeze(1)

        if self.is_instance:
            return logits, instance_logits, x_cls, x_seq, attn_layer1, attn_layer2
        else:
            return logits, instance_logits
        
class AmbiguousMILwithCL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,
                 dropout=0., is_instance=False, context_kernel_size=7):
        super().__init__()
        self.is_instance = is_instance
     
        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features
        )
        
        # define WPE    
        self.wave1 = torch.randn(2, mDim, 1)
        self.wave1[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, mDim, 1)
        self.wave2[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, mDim, 1)
        self.wave3[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave3 = nn.Parameter(self.wave3)    
            
        self.wave1_ = torch.randn(2, mDim, 1)
        self.wave1_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave1_ = nn.Parameter(self.wave1_)
        
        self.wave2_ = torch.zeros(2, mDim, 1)
        self.wave2_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave2_ = nn.Parameter(self.wave2_)
        
        self.wave3_ = torch.zeros(2, mDim, 1)
        self.wave3_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale > 0
        self.wave3_ = nn.Parameter(self.wave3_)        

        hidden_len = 2 * max_seq_len
            
        # define class token      
        self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim))  # (1, C, D)
        self.pos_layer = WaveletEncoding(mDim, max_seq_len, hidden_len) 
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len) 

        self.layer1 = TransLayer_amb(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer_amb(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)

        # bag-level head
        self._fc2 = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )
        
        self.cls_reduce = nn.Conv1d(
            in_channels=mDim, out_channels=1, kernel_size=1, bias=True
        ) # TODO Average pooling으로 바꿀까 고민중
        # self.cls_reduce = nn.AdaptiveAvgPool1d(1)

        # instance-level head (per-timestep)
        self.instance_head = nn.Sequential(
            # nn.Linear(mDim, mDim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )

        # 🔹 context encoder: local window를 보는 1D conv
        #   입력: [B, T, D] -> transpose -> [B, D, T]
        #   kernel_size = context_kernel_size (홀수 추천), padding으로 same length 유지
        self.context_kernel_size = context_kernel_size
        padding = context_kernel_size // 2
        self.context_encoder = nn.Conv1d(
            in_channels=mDim,
            out_channels=mDim,  # context feature 차원 D와 동일하게
            kernel_size=context_kernel_size,
            padding=padding,
            bias=False,
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.n_classes = n_classes
        
        initialize_weights(self)
        
        
    def forward(self, x, warmup=False):
        # x: [B, T, in_features]
        # feature extractor는 [B, C, T] -> [B, D, T] 형태라고 가정
        x1 = self.feature_extractor(x.transpose(1, 2))  # -> [B, D, T]
        x1 = x1.transpose(1, 2)  # -> [B, T, D]
        x = x1

        B, seq_len, D = x.shape
        C = self.n_classes        

        # cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, C, D]
      
        # concat cls tokens + sequence
        x = torch.cat((cls_tokens, x), dim=1)            # [B, C+T, D]

        # WPE1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3, self.n_classes)
        # TransLayer 1
        x = self.layer1(x)
        attn_layer1 = None  # attention weights are not used in SDPA
        # WPE2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_, self.n_classes)
        
        # TransLayer 2
        x = self.layer2(x)
        attn_layer2 = None  # attention weights are not used in SDPA
        # cls token branch
        x_cls = x[:, :C, :]                 # [B, C, D]
        x_cls_chfirst = x_cls.permute(0, 2, 1)  # [B, D, C]

        # sequence branch (per-timestep features)
        x_seq = x[:, C:, :]                 # [B, T, D]  -> 이게 z_t

        # 🔹 local-context representation c_t (context encoder)
        #    [B, T, D] -> [B, D, T] -> conv1d -> [B, D, T] -> [B, T, D]
        x_seq_chfirst = x_seq.transpose(1, 2)         # [B, D, T]
        c_seq_chfirst = self.context_encoder(x_seq_chfirst)  # [B, D, T]
        c_seq = c_seq_chfirst.transpose(1, 2)         # [B, T, D]  -> 이게 c_t

        # instance logits (per-timestep classification head)
        instance_logits = self.instance_head(x_seq)   # [B, T, C]

        # bag logits (from class tokens)
        logits = self.cls_reduce(x_cls_chfirst).squeeze(1)  # [B, C]

        # 🔹 is_instance=True일 때, z_t 와 c_t 둘 다 넘겨서 contrast loss에서 사용
        if self.is_instance:
            # logits: [B, C]
            # instance_logits: [B, T, C]
            # x_cls: [B, C, D], x_seq: [B, T, D], c_seq: [B, T, D]
            # attn_layer1, attn_layer2: 그대로 (visualization이나 auxiliary loss에 사용)
            return logits, instance_logits, x_cls, x_seq, c_seq, attn_layer1, attn_layer2
        else:
            return logits, instance_logits