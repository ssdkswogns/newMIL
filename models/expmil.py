import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inceptiontime import InceptionTimeFeatureExtractor

# 1. Query(Class)가 Key/Value(Sequence)를 조회하는 Cross Attention 모듈 정의
class ClassQueryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query는 Class Token에서 오고, KV는 Sequence에서 옴
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x_cls, x_seq):
        """
        x_cls: [B, n_classes, D] (Query) - 우리가 학습시킬 Prototype
        x_seq: [B, seq_len, D]   (Key, Value) - Inception Feature
        """
        B, C, D = x_cls.shape
        _, T, _ = x_seq.shape
        
        # 1. Cross Attention (Pre-Norm style)
        q = self.q_proj(self.norm(x_cls))  # [B, C, D]
        k = self.k_proj(x_seq)             # [B, T, D]
        v = self.v_proj(x_seq)             # [B, T, D]

        # Multi-head split
        q = q.reshape(B, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, C, d]
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # [B, H, d, T] (Transposed)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, T, d]

        # Scaled Dot-Product Attention
        attn = (q @ k) * self.scale  # [B, H, C, T] -> 각 Class가 어떤 Time step을 중요하게 보는지
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Combine
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, C, D) # [B, H, C, d] -> [B, C, D]
        
        # Residual Connection
        x_cls = x_cls + self.proj(out)
        
        # 2. Feed Forward Network
        x_cls = x_cls + self.ffn(self.norm_ffn(x_cls))
        
        return x_cls, attn


# class AmbiguousMILwithCL(nn.Module):
#     def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,
#                  dropout=0., is_instance=False, context_kernel_size=7):
#         super().__init__()
#         self.is_instance = is_instance
#         self.n_classes = n_classes
#         self.mDim = mDim

#         # 1. Feature Extractor (기존 유지)
#         self.feature_extractor = InceptionTimeFeatureExtractor(
#             n_in_channels=in_features
#         )
        
#         # 2. Learnable Queries (Class Prototypes)
#         # Bag 전체를 대표하는 각 Class별 Prototype 초기값
#         self.cls_token = nn.Parameter(torch.randn(1, n_classes, mDim))

#         # 3. Cross Attention Layer (Transformer Stack 대체)
#         # Class Token이 Sequence 전체를 훑어서 정보를 가져오는 역할
#         self.cross_attn = ClassQueryAttention(dim=mDim, num_heads=4, dropout=dropout)

#         # 4. Bag-level Classifier Head
#         # x_cls [B, C, D] -> Logits [B, C]를 만들기 위해 D차원을 1차원으로 축소
#         self.cls_reduce = nn.Conv1d(
#             in_channels=mDim, out_channels=1, kernel_size=1, bias=True
#         )

#         # 5. Instance-level Head (기존 유지)
#         self.instance_head = nn.Sequential(
#             nn.Linear(mDim, n_classes)
#         )
#         # self.contrastive_projector = nn.Sequential(
#         #     nn.Linear(mDim, mDim),
#         #     nn.ReLU(),
#         #     nn.Linear(mDim, mDim)
#         # )
#         self.contrastive_projector = nn.Identity()

#         # 6. Context Encoder (기존 유지)
#         self.context_kernel_size = context_kernel_size
#         padding = context_kernel_size // 2

#         # Weight Initialization
#         self._init_weights()

#     def _init_weights(self):
#         # 파라미터 초기화 (안정적인 학습을 위해 중요)
#         nn.init.xavier_normal_(self.cls_token)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x, warmup=False):
#         # x: [B, T, in_features]
        
#         # 1. Extract Features
#         # Feature extractor 가정: [B, C_in, T] -> [B, D, T]
#         x_feat = self.feature_extractor(x.transpose(1, 2)) 
#         x_seq = x_feat.transpose(1, 2)  # [B, T, D] -> 이것이 Key, Value가 됨

#         B, T, D = x_seq.shape

#         # 2. Instance Branch (Sequence 자체에 대한 처리)
#         # Instance Logits 계산
#         instance_logits = self.instance_head(x_seq)   # [B, T, C]
        
#         c_seq = None

#         # 3. Bag Branch (Class Prototypes 추출)
#         # Learnable Query 확장
#         cls_tokens = self.cls_token.expand(B, -1, -1) # [B, C, D]
        
#         # Cross Attention: Query(Class)가 Sequence(Key/Value)를 Attention
#         # x_cls: 업데이트된 Class Prototype [B, C, D]
#         # attn_weights: [B, H, C, T] (시각화 가능: 어떤 Time step이 중요한지)
#         x_cls, attn_weights = self.cross_attn(x_cls=cls_tokens, x_seq=x_seq)
        
#         # 4. Bag Classification
#         # [B, C, D] -> Permute -> [B, D, C] -> Conv1d(1x1) -> [B, 1, C] -> Squeeze -> [B, C]
#         x_cls_chfirst = x_cls.permute(0, 2, 1)
#         logits = self.cls_reduce(x_cls_chfirst).squeeze(1)

#         x_seq_proj = self.contrastive_projector(x_seq.reshape(-1, D))
#         x_seq_proj = x_seq_proj.reshape(B, T, D)
        
#         # x_cls: [B, C, D] -> [B*C, D]로 펼쳐서 투영
#         B, C, D = x_cls.shape
#         x_cls_proj = self.contrastive_projector(x_cls.reshape(-1, D))
#         x_cls_proj = x_cls_proj.reshape(B, C, D)

#         if self.is_instance:
#             return logits, instance_logits, x_cls, x_seq, c_seq, attn_weights, x_cls_proj, x_seq_proj
#         else:
#             return logits, instance_logits
        
class AmbiguousMILwithCL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,
                 dropout=0., is_instance=False, context_kernel_size=7):
        super().__init__()
        self.is_instance = is_instance
        self.n_classes = n_classes
        self.mDim = mDim

        # 1. Feature Extractor
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features
        )

        # 2. Instance-level Head: x_seq -> instance logits
        self.instance_head = nn.Linear(mDim, n_classes)

        # 3. Per-class temporal attention head
        #    x_seq[b,t,:] (D) -> score[b,t,c] (C)
        self.attention_head = nn.Sequential(
            nn.Linear(mDim, 2 * mDim),
            nn.Tanh(),
            nn.Linear(2 * mDim, n_classes)
        )

        self.instance_attn = nn.Sequential(
            nn.Linear(mDim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, warmup=False):
        # x: [B, T, in_features]

        # 1. InceptionTime backbone
        x_feat = self.feature_extractor(x.transpose(1, 2))   # [B, D, T]
        x_seq = x_feat.transpose(1, 2)                       # [B, T, D]

        B, T, D = x_seq.shape

        # 2. Instance branch
        instance_logits = self.instance_head(x_seq)          # [B, T, C]

        attn_seq = self.instance_attn(x_seq)
        weighted_instance_logits = instance_logits * attn_seq
        logits = torch.mean(weighted_instance_logits, dim=1)

        # 3. Bag-level logits (GAP over time in logit space)
        # logits = instance_logits.mean(dim=1)                 # [B, C]

        # 4. Per-class attention: x_seq -> score -> softmax over time
        #    score: [B, T, C]
        score = self.attention_head(x_seq)                   # [B, T, C]
        score = score.permute(0, 2, 1)                       # [B, C, T]

        attn = F.softmax(score, dim=-1)                      # [B, C, T]

        # 5. Attention-weighted class prototypes x_cls: [B, C, D]
        #    batched matmul: (B, C, T) x (B, T, D) -> (B, C, D)
        x_cls = torch.bmm(attn, x_seq)                       # [B, C, D]
        logits = x_cls.mean(dim=-1)                           # [B, C]

        # 6. Contrastive projections (for proto CL)
        x_seq_proj = None
        x_cls_proj = None

        # train() / test() 쪽 인터페이스 유지
        c_seq = None
        attn_weights = attn   # [B, C, T]

        if self.is_instance:
            return logits, instance_logits, x_cls, x_seq, c_seq, attn_weights, x_cls_proj, x_seq_proj
        else:
            return logits, instance_logits