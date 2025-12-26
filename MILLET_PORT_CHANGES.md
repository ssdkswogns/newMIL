# MILLET 모델 이식 상세 문서

## 📋 목차
1. [개요](#1-개요)
2. [아키텍처](#2-아키텍처)
3. [구현 세부사항](#3-구현-세부사항)
4. [사용 방법](#4-사용-방법)
5. [TimeMIL vs MILLET](#5-timemil-vs-millet)
6. [문제 해결](#6-문제-해결)

---

## 1. 개요

### 1.1 이식 배경
Amazon의 [MILTimeSeriesClassification](https://github.com/amazon-science/MILTimeSeriesClassification) 프로젝트에서 **MILLET (Multiple Instance Learning for Long-horizon timE series classificaTion)** 모델을 현재 프로젝트로 성공적으로 이식하였습니다.

**원본 소스:**
- 저장소: [amazon-science/MILTimeSeriesClassification](https://github.com/amazon-science/MILTimeSeriesClassification)
- 원본 파일: `/home/moon/code/MILTimeSeriesClassification/millet/model/millet_model.py`
- 라이선스: Apache-2.0

### 1.2 이식 목표
- ✅ MILLET 모델을 기존 TimeMIL 프레임워크에 통합
- ✅ 5가지 MIL pooling 방식 지원 (conjunctive, attention, instance, additive, gap)
- ✅ AOPCR (Area Over Perturbation Curve for Randomization) 해석 가능성 메트릭 지원
- ✅ 기존 학습/평가 파이프라인과 완벽한 호환성
- ✅ DDP (Distributed Data Parallel) 지원

---

## 2. 아키텍처

### 2.1 전체 구조

```
입력 [B, T, D]
    ↓ transpose(1,2)
[B, D, T]
    ↓
InceptionTime Feature Extractor
    ↓
[B, backbone_out_dim, T]  (backbone_out_dim = out_channels * 4)
    ↓
Feature Projection (1x1 Conv)
    ↓
[B, mDim, T]
    ↓ transpose(1,2)
[B, T, mDim]
    ↓
MIL Pooling Layer (Positional Encoding + Dropout)
    ↓
출력: bag_logits [B, C]
      instance_logits [B, C, T] (선택적)
      interpretation [B, C, T] (선택적)
      attention [B, T, 1] (선택적)
```

### 2.2 주요 컴포넌트

#### 2.2.1 InceptionTime Backbone
**파일**: `models/inceptiontime.py`

```python
InceptionTimeFeatureExtractor(
    n_in_channels=feats_size,  # 입력 feature 차원
    out_channels=max(1, mDim // 4),  # 출력 채널 (mDim의 1/4)
    padding_mode='replicate',  # Padding 방식
)
```

**구조:**
- 2개의 InceptionBlock
  - 각 InceptionBlock: 3개의 InceptionModule
  - 각 InceptionModule:
    - Bottleneck (1x1 conv)
    - 3개의 parallel conv (kernel size: 10, 20, 40)
    - Max pooling branch
    - Concatenation → 4 * out_channels
  - Residual connection
  - ReLU activation

**특징:**
- Multi-scale feature extraction (다양한 시간 스케일의 패턴 포착)
- Residual connection으로 gradient flow 개선
- Batch normalization으로 안정적인 학습

#### 2.2.2 Feature Projection
```python
backbone_out_dim = out_channels * 4  # InceptionTime 출력 차원

if backbone_out_dim != mDim:
    # 차원이 다르면 1x1 conv로 projection
    self.feature_proj = nn.Conv1d(backbone_out_dim, mDim, kernel_size=1)
else:
    # 같으면 그냥 통과
    self.feature_proj = nn.Identity()
```

**목적:**
- Backbone 출력 차원을 원하는 embedding 차원(mDim)으로 조정
- 유연한 하이퍼파라미터 설정 가능

#### 2.2.3 Positional Encoding
```python
class PositionalEncoding(nn.Module):
    """Transformer 스타일의 sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        # PE[pos, 2i] = sin(pos / 10000^(2i/d_model))
        # PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

**특징:**
- 최대 5000 timestep 지원
- Learnable이 아닌 fixed encoding (parameter 절약)
- 선택적 position index 지원 (불규칙한 시간 간격 대응 가능)

### 2.3 MIL Pooling 방식 상세 설명

#### 2.3.1 Conjunctive Pooling (추천) 🌟
**코드**: `MILConjunctivePooling` (lines 226-266)

**개념:**
Conjunctive pooling은 각 timestep에서 독립적으로 클래스 예측을 수행한 후, attention 메커니즘으로 중요한 timestep을 선택하여 가중 평균을 계산합니다. "Conjunctive"라는 이름은 instance prediction과 attention을 결합한다는 의미입니다.

**수식:**

$$
\begin{align}
\text{Input: } & \mathbf{x} \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Attention: } & \mathbf{a}_t = \text{MLP}_{\text{attn}}(\mathbf{x}_t) \in \mathbb{R}^{B \times T \times 1} \\
& \mathbf{a}_t = \sigma(\mathbf{W}_2 \cdot \tanh(\mathbf{W}_1 \mathbf{x}_t + \mathbf{b}_1) + \mathbf{b}_2) \\
\\
\text{Instance Logits: } & \mathbf{y}_{\text{inst}} = \mathbf{W}_{\text{cls}} \mathbf{x}^T + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{B \times T \times C} \\
\\
\text{Weighted Logits: } & \tilde{\mathbf{y}}_{\text{inst}} = \mathbf{y}_{\text{inst}} \odot \mathbf{a}_t \in \mathbb{R}^{B \times T \times C} \\
\\
\text{Bag Prediction: } & \mathbf{y}_{\text{bag}} = \frac{1}{T} \sum_{t=1}^{T} \tilde{\mathbf{y}}_{\text{inst}}^{(t)} \in \mathbb{R}^{B \times C}
\end{align}
$$

**코드 구현:**

```python
class MILConjunctivePooling(MILPooling):
    def __init__(self, d_in: int, n_classes: int, d_attn: int = 8,
                 dropout: float = 0.1, apply_positional_encoding: bool = True):
        super().__init__(d_in, n_classes, dropout, apply_positional_encoding)

        # Attention head: x [B,T,D] → attn [B,T,1]
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),      # [B,T,D] → [B,T,d_attn]
            nn.Tanh(),                     # 비선형 활성화
            nn.Linear(d_attn, 1),          # [B,T,d_attn] → [B,T,1]
            nn.Sigmoid(),                  # 0~1 범위로 정규화
        )

        # Instance classifier: x [B,T,D] → logits [B,T,C]
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None):
        # instance_embeddings: [B, D, T] → transpose → [B, T, D]
        instance_embeddings = instance_embeddings.transpose(1, 2)

        # Positional Encoding 추가 (선택적)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)

        # Dropout 적용
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        # Attention 계산: [B, T, D] → [B, T, 1]
        attn = self.attention_head(instance_embeddings)

        # Instance logits 계산: [B, T, D] → [B, T, C]
        instance_logits = self.instance_classifier(instance_embeddings)

        # Attention으로 가중: [B, T, C] * [B, T, 1] → [B, T, C]
        weighted_instance_logits = instance_logits * attn

        # Bag prediction: mean over timesteps
        bag_logits = torch.mean(weighted_instance_logits, dim=1)  # [B, C]

        return {
            "bag_logits": bag_logits,                               # [B, C]
            "interpretation": weighted_instance_logits.transpose(1, 2),  # [B, C, T]
            "instance_logits": instance_logits.transpose(1, 2),     # [B, C, T]
            "attn": attn,                                           # [B, T, 1]
        }
```

**핵심 아이디어:**
1. **Instance-level 예측**: 각 timestep을 독립적으로 분류 → 클래스별 확률 제공
2. **Attention 가중치**: 어떤 timestep이 중요한지 학습
3. **가중 평균**: 중요한 timestep의 예측을 더 크게 반영

**장점:**
- 클래스별로 다른 중요도 맵 제공 (해석 가능성 ↑)
- Instance-level prediction으로 세밀한 예측 가능
- Attention으로 노이즈 timestep 억제

**단점:**
- 파라미터 수가 Instance/Attention pooling보다 많음
- 학습 시간이 약간 더 소요

**언제 사용?**
- ✅ 기본 선택지 (대부분의 경우 최고 성능)
- ✅ Instance-level supervision 없을 때
- ✅ 해석 가능성이 중요할 때 (클래스별 timestep 중요도)
- ✅ Multi-label 분류

#### 2.3.2 Attention Pooling
**코드**: `MILAttentionPooling` (lines 141-180)

**개념:**
Attention pooling은 embedding space에서 직접 attention 가중 평균을 수행한 후 분류합니다. 클래스별 중요도가 아닌, 전역적인 timestep 중요도를 학습합니다.

**수식:**

$$
\begin{align}
\text{Input: } & \mathbf{x} \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Attention: } & \mathbf{a}_t = \sigma(\mathbf{W}_2 \cdot \tanh(\mathbf{W}_1 \mathbf{x}_t + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^{B \times T \times 1} \\
\\
\text{Weighted Emb: } & \tilde{\mathbf{x}} = \mathbf{x} \odot \mathbf{a}_t \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Bag Embedding: } & \mathbf{z}_{\text{bag}} = \frac{1}{T} \sum_{t=1}^{T} \tilde{\mathbf{x}}^{(t)} \in \mathbb{R}^{B \times D} \\
\\
\text{Bag Prediction: } & \mathbf{y}_{\text{bag}} = \mathbf{W}_{\text{cls}} \mathbf{z}_{\text{bag}} + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{B \times C}
\end{align}
$$

**코드 구현:**

```python
class MILAttentionPooling(MILPooling):
    def __init__(self, d_in: int, n_classes: int, d_attn: int = 8,
                 dropout: float = 0.1, apply_positional_encoding: bool = True):
        super().__init__(d_in, n_classes, dropout, apply_positional_encoding)

        # Attention head: x [B,T,D] → attn [B,T,1]
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )

        # Bag classifier: z [B,D] → logits [B,C]
        self.bag_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None):
        # instance_embeddings: [B, D, T] → [B, T, D]
        instance_embeddings = instance_embeddings.transpose(1, 2)

        # Positional Encoding
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)

        # Dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        # Attention: [B, T, D] → [B, T, 1]
        attn = self.attention_head(instance_embeddings)

        # Weighted embeddings: [B, T, D] * [B, T, 1] → [B, T, D]
        instance_embeddings = instance_embeddings * attn

        # Bag embedding: mean over time
        bag_embedding = torch.mean(instance_embeddings, dim=1)  # [B, D]

        # Classification
        bag_logits = self.bag_classifier(bag_embedding)  # [B, C]

        return {
            "bag_logits": bag_logits,
            # Interpretation: attention repeated for all classes (same for all)
            "interpretation": attn.repeat(1, 1, self.n_classes).transpose(1, 2),  # [B, C, T]
            "attn": attn,  # [B, T, 1]
        }
```

**핵심 아이디어:**
1. **전역 중요도**: 클래스와 무관하게 전체적으로 중요한 timestep 찾기
2. **Embedding 가중 평균**: 중요한 timestep의 representation을 모아서 bag 표현
3. **후분류**: Bag 표현을 분류기에 입력

**장점:**
- 간단한 구조 (파라미터 적음)
- 학습 안정적
- 클래스 독립적 중요도로 일반화 성능 좋을 수 있음

**단점:**
- 클래스별 다른 중요도를 표현 못함
- Instance-level 예측 불가
- Interpretation이 모든 클래스에 동일 (제한적)

**언제 사용?**
- ✅ 간단한 baseline 필요
- ✅ 클래스 독립적인 중요도가 합리적인 경우
- ✅ 파라미터/메모리 제약

#### 2.3.3 Instance Pooling
**코드**: `MILInstancePooling` (lines 108-138)

**개념:**
Instance pooling은 가장 직관적인 MIL 방식입니다. 각 timestep(instance)을 독립적으로 분류한 후, 모든 예측을 평균냅니다. Attention이 없어 모든 timestep을 동등하게 취급합니다.

**수식:**

$$
\begin{align}
\text{Input: } & \mathbf{x} \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Instance Logits: } & \mathbf{y}_{\text{inst}}^{(t)} = \mathbf{W}_{\text{cls}} \mathbf{x}^{(t)} + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{B \times C}, \quad \forall t \in [1, T] \\
\\
\text{Bag Prediction: } & \mathbf{y}_{\text{bag}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{y}_{\text{inst}}^{(t)} \in \mathbb{R}^{B \times C}
\end{align}
$$

**MIL 가정과의 관계:**
- **Standard MIL (OR)**: $P(\text{bag} = 1) = 1 - \prod_{t=1}^{T} (1 - P(\text{inst}_t = 1))$
- **Instance pooling (평균)**: $P(\text{bag} = 1) \approx \frac{1}{T} \sum_{t=1}^{T} P(\text{inst}_t = 1)$
  - Softmax/Sigmoid 후 평균은 OR 연산의 근사

**코드 구현:**

```python
class MILInstancePooling(MILPooling):
    def __init__(self, d_in: int, n_classes: int,
                 dropout: float = 0.1, apply_positional_encoding: bool = True):
        super().__init__(d_in, n_classes, dropout, apply_positional_encoding)

        # Instance classifier: x [B,T,D] → logits [B,T,C]
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None):
        # instance_embeddings: [B, D, T] → [B, T, D]
        instance_embeddings = instance_embeddings.transpose(1, 2)

        # Positional Encoding
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)

        # Dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        # Instance-level classification: [B, T, D] → [B, T, C]
        instance_logits = self.instance_classifier(instance_embeddings)

        # Bag-level prediction: average over instances
        bag_logits = instance_logits.mean(dim=1)  # [B, C]

        return {
            "bag_logits": bag_logits,                           # [B, C]
            "interpretation": instance_logits.transpose(1, 2),  # [B, C, T]
            "instance_logits": instance_logits.transpose(1, 2), # [B, C, T]
        }
```

**핵심 아이디어:**
1. **동등한 투표**: 모든 timestep이 동등하게 bag label에 기여
2. **Instance-level supervision 가능**: Instance label이 있으면 직접 학습 가능
3. **확률 평균**: Logit 평균은 확률 평균의 근사

**장점:**
- 가장 간단한 구조 (attention 없음)
- Instance-level label 활용 가능
- 해석 용이 (각 timestep의 예측 그대로)
- 파라미터 최소 (classifier만)

**단점:**
- 노이즈 timestep도 동등하게 반영 (성능 저하 가능)
- 중요한 timestep 강조 불가
- Attention pooling보다 성능 낮을 수 있음

**언제 사용?**
- ✅ Instance-level label이 있을 때
- ✅ 모든 timestep이 비슷하게 중요할 때
- ✅ 가장 간단한 baseline 필요
- ✅ Timestep별 확률 분포 분석 필요

#### 2.3.4 Additive Pooling
**코드**: `MILAdditivePooling` (lines 183-223)

**개념:**
Additive pooling은 먼저 embedding에 attention을 곱한 후 instance classification을 수행합니다. Conjunctive와 순서만 다르지만, 다른 inductive bias를 가집니다.

**수식:**

$$
\begin{align}
\text{Input: } & \mathbf{x} \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Attention: } & \mathbf{a}_t = \sigma(\mathbf{W}_2 \cdot \tanh(\mathbf{W}_1 \mathbf{x}_t + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^{B \times T \times 1} \\
\\
\text{Weighted Emb: } & \tilde{\mathbf{x}} = \mathbf{x} \odot \mathbf{a}_t \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Instance Logits: } & \mathbf{y}_{\text{inst}} = \mathbf{W}_{\text{cls}} \tilde{\mathbf{x}}^T + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{B \times T \times C} \\
\\
\text{Bag Prediction: } & \mathbf{y}_{\text{bag}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{y}_{\text{inst}}^{(t)} \in \mathbb{R}^{B \times C}
\end{align}
$$

**Conjunctive vs Additive 비교:**

| 순서 | Conjunctive | Additive |
|-----|-------------|----------|
| 1단계 | Instance logits 계산 | Attention 계산 |
| 2단계 | Attention 계산 | Embedding 가중 |
| 3단계 | Logits × Attention | Instance logits 계산 (가중된 emb 사용) |
| 4단계 | 평균 | 평균 |

**코드 구현:**

```python
class MILAdditivePooling(MILPooling):
    def __init__(self, d_in: int, n_classes: int, d_attn: int = 8,
                 dropout: float = 0.1, apply_positional_encoding: bool = True):
        super().__init__(d_in, n_classes, dropout, apply_positional_encoding)

        # Attention head
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )

        # Instance classifier (applied to weighted embeddings)
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None):
        # instance_embeddings: [B, D, T] → [B, T, D]
        instance_embeddings = instance_embeddings.transpose(1, 2)

        # Positional Encoding
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)

        # Dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        # Attention: [B, T, D] → [B, T, 1]
        attn = self.attention_head(instance_embeddings)

        # Weight embeddings first: [B, T, D] * [B, T, 1] → [B, T, D]
        instance_embeddings = instance_embeddings * attn

        # Then classify: [B, T, D] → [B, T, C]
        instance_logits = self.instance_classifier(instance_embeddings)

        # Average
        bag_logits = instance_logits.mean(dim=1)  # [B, C]

        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * attn).transpose(1, 2),  # [B, C, T]
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }
```

**핵심 아이디어:**
1. **Embedding 필터링**: Attention으로 중요한 feature만 살림
2. **가중된 representation 분류**: 필터링된 embedding으로 예측
3. **Additive 효과**: Attention이 embedding 차원에서 작용

**장점:**
- Attention이 embedding level에서 작용 (다른 inductive bias)
- Conjunctive와 비슷한 표현력
- 클래스별 중요도 제공

**단점:**
- Conjunctive와 성능 유사 (데이터셋에 따라 다름)
- 약간 더 복잡한 해석

**언제 사용?**
- ✅ Conjunctive 대안으로 실험
- ✅ Embedding-level attention이 더 합리적인 경우
- ✅ 다른 inductive bias 시도

#### 2.3.5 Global Average Pooling (GAP)
**코드**: `GlobalAveragePooling` (lines 69-105)

**개념:**
GAP은 EmbeddingSpace MIL이라고도 불리며, 가장 간단한 baseline입니다. Positional encoding이나 attention 없이 단순히 시간 축으로 평균을 낸 후 분류합니다. CAM (Class Activation Mapping)으로 해석 가능성을 제공합니다.

**수식:**

$$
\begin{align}
\text{Input: } & \mathbf{x} \in \mathbb{R}^{B \times T \times D} \\
\\
\text{Bag Embedding: } & \mathbf{z}_{\text{bag}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{x}^{(t)} \in \mathbb{R}^{B \times D} \\
\\
\text{Bag Prediction: } & \mathbf{y}_{\text{bag}} = \mathbf{W}_{\text{cls}} \mathbf{z}_{\text{bag}} + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{B \times C} \\
\\
\text{CAM (Interpretation): } & \text{CAM}_{c,t} = \mathbf{W}_{\text{cls}}[c, :] \cdot \mathbf{x}^{(t)} \in \mathbb{R}^{B \times C \times T}
\end{align}
$$

**CAM (Class Activation Map):**
- Classifier weight $\mathbf{W}_{\text{cls}} \in \mathbb{R}^{C \times D}$와 각 timestep embedding $\mathbf{x}^{(t)}$의 내적
- 각 클래스 $c$에 대해 timestep $t$가 얼마나 기여하는지 측정
- Gradient 없이 계산 가능한 해석 방법

**코드 구현:**

```python
class GlobalAveragePooling(MILPooling):
    def __init__(self, d_in: int, n_classes: int,
                 dropout: float = 0.0, apply_positional_encoding: bool = False):
        # GAP은 PE와 dropout을 사용하지 않음 (가장 간단한 형태)
        super().__init__(d_in, n_classes, dropout=dropout,
                        apply_positional_encoding=apply_positional_encoding)

        # Bag classifier only
        self.bag_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None):
        # instance_embeddings: [B, D, T]

        # (선택적) Positional Encoding - 기본적으로 사용 안 함
        instance_embeddings_transposed = instance_embeddings.transpose(1, 2)  # [B, T, D]
        if self.apply_positional_encoding:
            instance_embeddings_transposed = self.positional_encoding(
                instance_embeddings_transposed, pos
            )
        if self.dropout_p > 0:
            instance_embeddings_transposed = self.dropout(instance_embeddings_transposed)
        instance_embeddings = instance_embeddings_transposed.transpose(2, 1)  # [B, D, T]

        # CAM 계산: W [C, D] @ x [B, D, T] → [B, C, T]
        cam = self.bag_classifier.weight @ instance_embeddings  # Broadcasting

        # Bag embedding: 시간 축 평균
        bag_embeddings = instance_embeddings.mean(dim=-1)  # [B, D]

        # Classification
        bag_logits = self.bag_classifier(bag_embeddings)  # [B, C]

        return {
            "bag_logits": bag_logits,  # [B, C]
            "interpretation": cam,     # [B, C, T] - CAM
        }
```

**핵심 아이디어:**
1. **단순 평균**: 복잡한 메커니즘 없이 전체 시퀀스 표현
2. **CAM 해석**: Classifier weight로 각 timestep의 기여도 계산
3. **효율성**: 최소 파라미터, 최소 연산

**장점:**
- 가장 빠른 학습/추론
- 가장 적은 메모리 사용
- 가장 적은 파라미터 (classifier만)
- Gradient 없이 해석 가능 (CAM)
- 구현 간단

**단점:**
- 성능이 다른 방식보다 낮을 수 있음
- 중요 timestep 선택 불가
- Positional information 사용 안 함
- CAM 해석의 정확도가 attention보다 낮을 수 있음

**언제 사용?**
- ✅ 가장 빠른 baseline 필요
- ✅ 메모리 심각하게 제약
- ✅ 전체 시퀀스 평균적 특징이 충분한 경우
- ✅ 단순성이 중요

### 2.4 Pooling 방식 비교표

| Pooling | Attention 사용 | Instance Logits | Interpretation | PE | Params | 복잡도 |
|---------|---------------|-----------------|----------------|-----|--------|--------|
| Conjunctive | ✓ | ✓ | Weighted inst. | ✓ | Medium | Medium |
| Attention | ✓ | ✗ | Attention | ✓ | Low | Low |
| Instance | ✗ | ✓ | Instance logits | ✓ | Low | Low |
| Additive | ✓ | ✓ | Weighted inst. | ✓ | Medium | Medium |
| GAP | ✗ | ✗ | CAM | ✗ | Lowest | Lowest |

---

## 2.5 평가 메트릭 (Evaluation Metrics)

MILLET 모델의 성능은 다음과 같은 메트릭으로 평가됩니다.

### 2.5.1 분류 성능 메트릭

#### **1. Accuracy (정확도)**

$$
\text{Accuracy} = \frac{\text{올바르게 분류된 샘플 수}}{\text{전체 샘플 수}} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**설명:**
- 전체 예측 중 올바른 예측의 비율
- 클래스 불균형 시 misleading할 수 있음

**코드:**
```python
from sklearn.metrics import accuracy_score

# Bag-level prediction
_, pred_classes = torch.max(bag_logits, dim=1)  # [B]
accuracy = accuracy_score(true_labels, pred_classes)
```

---

#### **2. Balanced Accuracy (균형 정확도)**

$$
\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}
$$

**설명:**
- 각 클래스의 recall을 평균
- 클래스 불균형 데이터셋에서 더 공정한 평가

**코드:**
```python
from sklearn.metrics import balanced_accuracy_score

bal_acc = balanced_accuracy_score(true_labels, pred_classes)
```

---

#### **3. AUROC (Area Under ROC Curve)**

$$
\text{AUROC} = \int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))
$$

where:
- TPR (True Positive Rate) = Recall = $\frac{TP}{TP + FN}$
- FPR (False Positive Rate) = $\frac{FP}{FP + TN}$

**설명:**
- ROC 곡선 아래 면적
- 임계값과 무관한 분류기 성능 측정
- 0.5 = random, 1.0 = perfect

**Multi-class 확장:**
- **OvO (One-vs-One)**: 모든 클래스 쌍에 대해 AUROC 계산 후 평균
- **OvR (One-vs-Rest)**: 각 클래스를 나머지와 비교하여 AUROC 계산 후 평균
- **Macro**: 클래스별 AUROC의 산술 평균
- **Weighted**: 클래스 크기로 가중 평균

**코드:**
```python
from sklearn.metrics import roc_auc_score

# Single-label (multi-class)
prob = torch.softmax(bag_logits, dim=1).cpu().numpy()
auroc = roc_auc_score(true_labels, prob, multi_class='ovo', average='weighted')

# Multi-label
prob = torch.sigmoid(bag_logits).cpu().numpy()
auroc = roc_auc_score(true_labels, prob, average='macro')
```

---

#### **4. mAP (mean Average Precision)**

**Average Precision (AP)** for class $c$:

$$
\text{AP}_c = \sum_{k=1}^{n} P(k) \Delta R(k)
$$

where $P(k)$ = precision at rank $k$, $\Delta R(k)$ = change in recall

**mean Average Precision (mAP)**:

$$
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
$$

**설명:**
- Precision-Recall 곡선 아래 면적
- Multi-label 분류에서 특히 유용
- 클래스 불균형에 robust

**코드:**
```python
from sklearn.metrics import average_precision_score

# Multi-label
prob = torch.sigmoid(bag_logits).cpu().numpy()
mAP = average_precision_score(true_labels, prob, average='macro')

# Per-class AP
ap_per_class = average_precision_score(true_labels, prob, average=None)
```

---

#### **5. F1 Score**

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}
$$

**Macro F1** (클래스별 F1 평균):

$$
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c
$$

**설명:**
- Precision과 Recall의 조화 평균
- 불균형 데이터에서 accuracy보다 유용

**코드:**
```python
from sklearn.metrics import f1_score

# Binary or multi-class
f1_macro = f1_score(true_labels, pred_classes, average='macro')
f1_weighted = f1_score(true_labels, pred_classes, average='weighted')
```

---

### 2.5.2 해석 가능성 메트릭

#### **6. AOPCR (Area Over Perturbation Curve for Randomization)**

**개념:**
모델의 설명(interpretation)이 얼마나 신뢰할 수 있는지 측정하는 메트릭입니다.

**절차:**
1. 모델의 interpretation으로 timestep 중요도 추출
2. 중요도 높은 timestep부터 순차적으로 제거 (0으로 마스킹)
3. 각 제거 비율 $\alpha$에서 logit 하락 측정
4. Random baseline과 비교

**수식:**

$$
\begin{align}
\text{Drop}_{\text{expl}}(\alpha) &= \text{logit}_{\text{orig}} - \text{logit}_{\text{perturbed}}(\alpha, \text{explanation}) \\
\text{Drop}_{\text{rand}}(\alpha) &= \text{logit}_{\text{orig}} - \text{logit}_{\text{perturbed}}(\alpha, \text{random}) \\
\\
\text{AOPC}_{\text{expl}} &= \frac{1}{N} \sum_{i=1}^{N} \text{Drop}_{\text{expl}}(\alpha_i) \\
\text{AOPC}_{\text{rand}} &= \frac{1}{N} \sum_{i=1}^{N} \text{Drop}_{\text{rand}}(\alpha_i) \\
\\
\text{AOPCR} &= \text{AOPC}_{\text{expl}} - \text{AOPC}_{\text{rand}}
\end{align}
$$

**해석:**
- **AOPCR > 0**: 모델 설명이 random보다 좋음 ✅
- **AOPCR ≈ 0**: 설명이 random과 비슷 (문제) ⚠️
- **AOPCR < 0**: 설명이 random보다 나쁨 (심각) ❌

**코드 예시:**
```python
def compute_aopcr(model, x, target_class, alphas=[0.0, 0.05, 0.1, ..., 0.5]):
    # 원본 logit
    orig_logit = model(x)["bag_logits"][0, target_class].item()

    # Interpretation 추출
    interpretation = model(x)["interpretation"][0, target_class]  # [T]
    sorted_idx = torch.argsort(interpretation, descending=True)  # 중요도 순

    drops_expl = []
    drops_rand = []

    for alpha in alphas:
        k = int(alpha * T)  # 제거할 timestep 수

        # Explanation-based removal
        x_perturbed = x.clone()
        x_perturbed[:, sorted_idx[:k], :] = 0.0
        logit_expl = model(x_perturbed)["bag_logits"][0, target_class].item()
        drops_expl.append(orig_logit - logit_expl)

        # Random removal (n_random회 평균)
        rand_drops = []
        for _ in range(n_random):
            rand_idx = torch.randperm(T)[:k]
            x_perturbed_rand = x.clone()
            x_perturbed_rand[:, rand_idx, :] = 0.0
            logit_rand = model(x_perturbed_rand)["bag_logits"][0, target_class].item()
            rand_drops.append(orig_logit - logit_rand)
        drops_rand.append(np.mean(rand_drops))

    aopc_expl = np.mean(drops_expl)
    aopc_rand = np.mean(drops_rand)
    aopcr = aopc_expl - aopc_rand

    return aopcr
```

**MILLET에서의 적용:**
- **Conjunctive/Additive/Instance**: `interpretation` 사용 (클래스별 timestep 중요도)
- **Attention**: `attn` 사용 (모든 클래스 동일)
- **GAP**: CAM 사용

---

#### **7. NDCG@N (Normalized Discounted Cumulative Gain)**

**개념:**
Instance-level ground truth가 있을 때, 모델이 중요한 instance를 상위에 랭킹하는지 측정합니다.

**수식:**

$$
\begin{align}
\text{DCG@N} &= \sum_{i=1}^{N} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)} \\
\\
\text{NDCG@N} &= \frac{\text{DCG@N}}{\text{IDCG@N}}
\end{align}
$$

where:
- $\text{rel}_i$ = relevance (중요도) of instance at rank $i$
- $\text{IDCG@N}$ = Ideal DCG (perfect ranking)

**코드:**
```python
def calculate_ndcg_at_n(importance_scores, instance_targets, n=None):
    """
    importance_scores: [T] - 모델이 예측한 각 timestep의 중요도
    instance_targets: [T] - Ground truth instance labels (0 or 1)
    """
    if n is None:
        n = len(instance_targets)

    # Rank by importance
    ranked_indices = torch.argsort(importance_scores, descending=True)
    ranked_relevance = instance_targets[ranked_indices[:n]]

    # DCG
    dcg = torch.sum((2**ranked_relevance - 1) / torch.log2(torch.arange(2, n+2, dtype=torch.float)))

    # IDCG (ideal: ground truth를 그대로 정렬)
    ideal_relevance = torch.sort(instance_targets, descending=True)[0][:n]
    idcg = torch.sum((2**ideal_relevance - 1) / torch.log2(torch.arange(2, n+2, dtype=torch.float)))

    if idcg == 0:
        return 0.0
    return (dcg / idcg).item()
```

---

### 2.5.3 평가 메트릭 선택 가이드

| 상황 | 추천 메트릭 |
|------|----------|
| **Balanced 데이터셋** | Accuracy, AUROC |
| **Imbalanced 데이터셋** | Balanced Accuracy, F1-Macro, mAP |
| **Multi-label 분류** | mAP, AUROC-Macro, F1-Macro |
| **Single-label 분류** | Accuracy, AUROC-OvO, F1-Weighted |
| **해석 가능성 평가** | AOPCR |
| **Instance-level GT 있음** | NDCG@N, Instance-level F1 |
| **임계값 튜닝 필요** | Precision-Recall Curve, F1 |
| **임계값 무관 평가** | AUROC, mAP |

---

### 2.5.4 main_cl_fix.py에서의 평가 구현

```python
# Bag-level 예측
prob = torch.sigmoid(bag_logits)  # [B, C]
_, pred_classes = torch.max(prob, dim=1)  # [B]

# 분류 메트릭
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, f1_score
)

acc = accuracy_score(y_true, pred_classes)
bal_acc = balanced_accuracy_score(y_true, pred_classes)

if args.datatype == 'original':  # Single-label
    auroc = roc_auc_score(y_true, prob.numpy(), multi_class='ovo', average='weighted')
else:  # Multi-label
    auroc = roc_auc_score(y_true, prob.numpy(), average='macro')
    mAP = average_precision_score(y_true, prob.numpy(), average='macro')

f1_macro = f1_score(y_true, pred_classes, average='macro')

print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"F1-Macro: {f1_macro:.4f}")
if args.datatype != 'original':
    print(f"mAP: {mAP:.4f}")

# AOPCR 계산 (compute_aopcr.py)
from compute_aopcr import compute_classwise_aopcr

aopcr_results = compute_classwise_aopcr(
    model, testloader, args,
    stop=0.5, step=0.05, n_random=3
)
aopcr_per_class, aopcr_weighted, aopcr_mean, aopcr_overall = aopcr_results[:4]

print(f"AOPCR (weighted): {aopcr_weighted:.4f}")
print(f"AOPCR (mean): {aopcr_mean:.4f}")
```

---

## 3. 구현 세부사항

### 3.1 파일 구조

```
newMIL/
├── models/
│   ├── milet.py          ⭐ 새로 생성 (380 lines)
│   ├── inceptiontime.py  (재사용)
│   ├── common.py         (재사용)
│   └── timemil.py        (기존)
├── main_cl_fix.py        ✏️ 수정 (MILLET 통합)
├── compute_aopcr.py      ✏️ 수정 (MILLET AOPCR 지원)
└── MILLET_PORT_CHANGES.md (이 문서)
```

### 3.2 models/milet.py 상세

**라인별 구조:**
- Lines 19-41: `PositionalEncoding`
- Lines 48-67: `MILPooling` (base class)
- Lines 69-105: `GlobalAveragePooling`
- Lines 108-138: `MILInstancePooling`
- Lines 141-180: `MILAttentionPooling`
- Lines 183-223: `MILAdditivePooling`
- Lines 226-266: `MILConjunctivePooling`
- Lines 273-380: `MILLET` (main model)
  - Line 368-373: is_instance=True 경로 (7개 반환값)
  - Line 375-377: is_instance=False + instance/interpretation 있을 때 (3개 반환값)
  - Line 379: is_instance=False + instance/interpretation 없을 때 (bag_logits만)

**MILLET 클래스 파라미터:**
```python
MILLET(
    feats_size: int,          # 입력 feature 차원 (e.g., 512)
    mDim: int = 128,          # Embedding 차원
    n_classes: int = 10,      # 클래스 개수
    dropout: float = 0.1,     # Dropout 비율
    max_seq_len: int = 256,   # 최대 시퀀스 길이 (사용 안 함, 호환성)
    pooling: str = 'conjunctive',  # Pooling 방식
    is_instance: bool = False,     # Instance-level 출력 활성화
    apply_positional_encoding: bool = True,  # PE 적용 여부
    padding_mode: str = 'replicate',  # Conv padding 모드
)
```

**Forward 출력 형식:**

```python
# Case 1: is_instance=True (평가 시) - 7개 반환값
if self.is_instance:
    x_seq = features.transpose(1, 2)  # [B, T, mDim]
    x_cls = x_seq.mean(dim=1, keepdim=True).expand(-1, self.n_classes, -1)  # [B, C, mDim]
    instance_pred = instance_logits.transpose(1, 2) if instance_logits is not None else None
    # Return tuple length 7 for compatibility with main_cl_fix.py
    return (bag_logits, instance_pred, x_cls, x_seq, interpretation, attn, attn)
    # bag_logits: [B, C]
    # instance_pred: [B, T, C] or None
    # x_cls: [B, C, mDim] (class embeddings - 호환성용)
    # x_seq: [B, T, mDim] (sequence embeddings)
    # interpretation: [B, C, T]
    # attn: attention weights (if available) or None (중복 반환)

# Case 2: is_instance=False (학습 시) - instance/interpretation 있으면 3개 반환
if instance_logits is not None or interpretation is not None:
    instance_pred = instance_logits.transpose(1, 2) if instance_logits is not None else None
    return (bag_logits, instance_pred, interpretation)
    # bag_logits: [B, C]
    # instance_pred: [B, T, C] or None
    # interpretation: [B, C, T] or None

# Case 3: is_instance=False이고 instance/interpretation 없으면 bag_logits만 반환
return bag_logits
```

### 3.3 main_cl_fix.py 통합

**변경사항 요약:**

1. **Import** (line 39)
```python
from models.milet import MILLET
```

2. **모델 생성** (lines 982-986)
```python
elif args.model == 'MILLET':
    base_model = MILLET(
        args.feats_size,           # 입력 차원
        mDim=args.embed,           # 128
        n_classes=num_classes,     # 데이터셋 클래스 수
        dropout=args.dropout_node, # 0.1
        max_seq_len=seq_len,       # 시퀀스 최대 길이 (호환성용)
        pooling=args.millet_pooling,  # conjunctive
        is_instance=False          # 학습 시에는 False
    ).to(device)
    proto_bank = None  # MILLET은 prototype bank 사용 안 함
```

3. **평가 모델 생성** (lines 1245-1248)
```python
elif args.model == 'MILLET':
    eval_model = MILLET(
        args.feats_size,
        mDim=args.embed,
        n_classes=args.num_classes,
        dropout=args.dropout_node,
        max_seq_len=args.seq_len,
        pooling=args.millet_pooling,
        is_instance=True  # 평가 시에는 True (해석 가능성 분석용, 7개 출력)
    ).to(device)
```

4. **Loss 계산 호환성** (lines 394-398)
```python
if isinstance(out, (tuple, list)):
    bag_prediction = out[0]
    # MILLET는 bag_loss만 사용하지만 eval을 위해 instance_pred를 유지
    instance_pred = out[1] if len(out) > 1 else None
else:
    bag_prediction = out
```

5. **평가 로직** (lines 570-574)
```python
elif args.model == 'MILLET':
    out = milnet(bag_feats)
    # MILLET도 eval시 instance_pred를 활용할 수 있게 유지
    instance_pred = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    attn_layer2 = None
```

6. **명령행 인수** (lines 786-790)
```python
parser.add_argument('--millet_pooling', default='conjunctive', type=str,
                    help="Pooling for MILLET (conjunctive | attention | instance | additive | gap)")
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model', default='TimeMIL', type=str,
                    help='MIL model (TimeMIL | newTimeMIL | AmbiguousMIL | MILLET)')
```

### 3.4 compute_aopcr.py MILLET 지원

**핵심 로직** (lines 150-204):

```python
elif args.model == 'MILLET':
    out = milnet(x)
    instance_logits = None
    interpretation = None

    # MILLET 출력 파싱
    if isinstance(out, (tuple, list)):
        logits = out[0]              # bag logits
        if len(out) > 1:
            instance_logits = out[1]  # instance predictions
        if len(out) > 2:
            interpretation = out[2]   # interpretation map
    else:
        logits = out

    # ...

    # Timestep 중요도 계산
    if interpretation is not None:
        scores = interpretation[b, pred_c]  # [T] - 우선순위 높음
    elif instance_logits is not None:
        scores = torch.softmax(instance_logits[b], dim=-1)[:, pred_c]
    else:
        raise ValueError("MILLET needs interpretation or instance logits for AOPCR")
```

**AOPCR 계산 과정:**
1. 원본 입력으로 bag logit 계산
2. Timestep 중요도 score 추출 (interpretation 또는 instance logits)
3. Score로 timestep 정렬
4. 상위 k% timestep을 0으로 마스킹
5. 마스킹된 입력으로 재예측
6. Logit 하락 정도 측정
7. Random baseline과 비교 → AOPCR

---

## 4. 사용 방법

### 4.1 기본 학습 (Single GPU)

```bash
python main_cl_fix.py \
  --dataset PenDigits \
  --datatype mixed \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --dropout_node 0.1 \
  --num_epochs 300 \
  --batchsize 64 \
  --lr 5e-3 \
  --weight_decay 1e-4 \
  --save_dir ./savemodel/
```

**주요 파라미터 설명:**
- `--dataset`: UCR/UEA 데이터셋 이름 (예: PenDigits, PAMAP2)
- `--datatype`:
  - `mixed`: Synthetic multi-label bags (MIL 실험용)
  - `original`: 원본 단일 시퀀스 데이터
- `--model MILLET`: MILLET 모델 사용
- `--millet_pooling`: conjunctive, attention, instance, additive, gap 중 선택
- `--embed`: mDim (embedding 차원)
- `--dropout_node`: Pooling layer의 dropout 비율

### 4.2 DDP (Multi-GPU) 학습

```bash
# 4 GPU 예시
torchrun --nproc_per_node=4 main_cl_fix.py \
  --dataset PenDigits \
  --datatype mixed \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --dropout_node 0.1 \
  --num_epochs 300 \
  --batchsize 64 \
  --lr 5e-3
```

**DDP 특징:**
- 자동으로 batch를 GPU별로 분할
- Rank 0 프로세스만 로깅/저장
- Gradient는 자동으로 all-reduce

### 4.3 다양한 Pooling 방식 실험

```bash
# Conjunctive (추천)
python main_cl_fix.py --model MILLET --millet_pooling conjunctive --embed 128

# Attention-based
python main_cl_fix.py --model MILLET --millet_pooling attention --embed 128

# Instance-level
python main_cl_fix.py --model MILLET --millet_pooling instance --embed 128

# Additive
python main_cl_fix.py --model MILLET --millet_pooling additive --embed 128

# Global Average Pooling (가장 빠름)
python main_cl_fix.py --model MILLET --millet_pooling gap --embed 128
```

### 4.4 체크포인트 저장 및 로드

**저장 경로:**
```
{save_dir}/InceptBackbone/{dataset}/exp_{n}/weights/
├── best_MILLET.pth         # 최고 성능 체크포인트
└── args.pkl                # 하이퍼파라미터 저장
```

**재학습:**
```bash
# 동일한 dataset/model/embed/pooling으로 실행하면
# 자동으로 같은 exp 폴더 찾아서 이어서 학습
python main_cl_fix.py \
  --dataset PenDigits \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --num_epochs 500  # epoch 수를 늘려서 재학습
```

### 4.5 평가 및 AOPCR 계산

**자동 평가:**
```bash
# 학습 완료 시 자동으로 best checkpoint 로드 후 평가
# Classification metrics + AOPCR 자동 계산
python main_cl_fix.py \
  --dataset PenDigits \
  --datatype mixed \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --num_epochs 300
```

**AOPCR만 실행:**
```bash
python compute_aopcr.py \
  --dataset PenDigits \
  --datatype mixed \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --feats_size 512
```

### 4.6 하이퍼파라미터 가이드

| 파라미터 | 기본값 | 권장 범위 | 설명 |
|---------|--------|----------|------|
| `--embed` | 128 | 64-256 | mDim (embedding 차원) |
| `--dropout_node` | 0.2 | 0.0-0.5 | Pooling layer dropout |
| `--lr` | 5e-3 | 1e-4 ~ 1e-2 | Learning rate |
| `--weight_decay` | 1e-4 | 1e-5 ~ 1e-3 | Weight decay |
| `--batchsize` | 64 | 16-128 | Batch size |
| `--num_epochs` | 300 | 200-500 | 학습 epoch |

**메모리 최적화:**
```bash
# OOM 발생 시
--embed 64 \
--batchsize 32 \
--millet_pooling gap  # 가장 가벼운 pooling
```

**성능 최적화:**
```bash
# 최고 성능을 위한 설정
--millet_pooling conjunctive \
--embed 128 \
--dropout_node 0.1 \
--lr 5e-3 \
--weight_decay 1e-4 \
--num_epochs 300
```

---

## 5. TimeMIL vs MILLET

### 5.1 상세 비교

| 특징 | TimeMIL | newTimeMIL | AmbiguousMIL | MILLET |
|-----|---------|-----------|-------------|--------|
| **Backbone** | Transformer | Transformer | Transformer | InceptionTime (CNN) |
| **Class Token** | 1개 | C개 | C개 | 없음 (pooling으로 대체) |
| **Positional Encoding** | Learnable | Learnable | Learnable | Sinusoidal (선택적) |
| **Self-Attention** | ✓ | ✓ | ✓ | ✗ |
| **Instance Prediction** | ✗ | ✗ | ✓ | Pooling 방식에 따라 |
| **Interpretation** | Attention | Attention | Instance logits | Pooling 방식에 따라 |
| **시간 복잡도** | O(T²) | O(T²) | O(T²) | O(T) |
| **공간 복잡도** | O(T²) | O(T²) | O(T²) | O(T) |
| **파라미터 수** | High | High | High | Low-Medium |
| **학습 속도** | 느림 | 느림 | 느림 | 빠름 |
| **추론 속도** | 느림 | 느림 | 느림 | 빠름 |
| **긴 시퀀스 (T>1000)** | 어려움 | 어려움 | 어려움 | 가능 |
| **Multi-scale Feature** | ✗ | ✗ | ✗ | ✓ (InceptionTime) |

### 5.2 성능 특성

**TimeMIL 장점:**
- Long-range dependency 모델링
- Attention visualization 직관적
- 전역적 맥락 파악 우수

**MILLET 장점:**
- 긴 시퀀스 효율적 처리 (O(T) vs O(T²))
- 메모리 효율적
- 빠른 학습/추론
- Multi-scale feature 자동 추출
- 다양한 pooling 선택지

### 5.3 사용 시나리오

#### MILLET 추천 상황 🌟
1. **긴 시퀀스** (T > 1000)
   - 예: 장기간 센서 데이터, 긴 행동 인식
2. **제한된 GPU 메모리**
   - 예: Single GPU, 낮은 VRAM
3. **빠른 실험 필요**
   - Baseline 빠르게 구축
4. **Multi-scale 패턴 중요**
   - 예: 다양한 시간 스케일의 특징
5. **Instance-level 해석 필요**
   - Pooling 선택으로 다양한 해석 가능

#### TimeMIL 추천 상황
1. **중간 길이 시퀀스** (T < 500)
2. **Long-range dependency 중요**
   - 전체 시퀀스 맥락이 핵심
3. **Class token 분석 필요**
   - Class embedding 연구
4. **Attention visualization 선호**

---

## 6. 문제 해결

### 6.1 에러 해결

#### Q1: "MILLET needs interpretation or instance logits for AOPCR"
```
원인: GAP pooling 사용 시 일부 설정에서 interpretation이 None일 수 있음
해결: conjunctive, attention, instance, additive 중 하나 사용
```

```bash
# 해결책
python main_cl_fix.py --model MILLET --millet_pooling conjunctive
```

#### Q2: CUDA Out of Memory (OOM)
```
원인: embed가 너무 크거나 batchsize가 큼
```

```bash
# 해결책 1: embed 줄이기
--embed 64

# 해결책 2: batchsize 줄이기
--batchsize 32

# 해결책 3: GAP pooling 사용 (가장 가벼움)
--millet_pooling gap

# 해결책 4: Gradient accumulation (구현 필요)
```

#### Q3: NaN Loss 발생
```
원인: Learning rate가 너무 높거나 dropout이 너무 높음
```

```bash
# 해결책
--lr 1e-3 \
--dropout_node 0.1 \
--weight_decay 1e-4
```

#### Q4: Dimension Mismatch
```
원인: feats_size 설정이 실제 데이터와 다름
```

```bash
# 데이터셋별 feats_size 확인
# PAMAP2: 512
# PenDigits: 데이터 로딩 시 자동 결정

# 해결책: 데이터 shape 출력 확인
print(f"Input shape: {x.shape}")  # [B, T, D]
# D가 feats_size
```

### 6.2 성능 튜닝

#### Pooling 방식별 성능 가이드

**Conjunctive (추천 🌟):**
- 대부분 데이터셋에서 최고 성능
- Instance-level supervision 없을 때 특히 좋음
```bash
--millet_pooling conjunctive --embed 128
```

**Attention:**
- 간단한 baseline
- 클래스 독립적 중요도
```bash
--millet_pooling attention --embed 128
```

**Instance:**
- Instance-level label 있을 때
- 명시적인 instance prediction 필요 시
```bash
--millet_pooling instance --embed 128
```

**GAP:**
- 빠른 baseline, 메모리 효율
- 간단한 데이터셋
```bash
--millet_pooling gap --embed 64
```

#### 하이퍼파라미터 조합 추천

**고성능 설정 (충분한 GPU 메모리):**
```bash
python main_cl_fix.py \
  --model MILLET \
  --millet_pooling conjunctive \
  --embed 128 \
  --dropout_node 0.1 \
  --batchsize 64 \
  --lr 5e-3 \
  --weight_decay 1e-4 \
  --num_epochs 300
```

**메모리 제약 설정:**
```bash
python main_cl_fix.py \
  --model MILLET \
  --millet_pooling gap \
  --embed 64 \
  --dropout_node 0.0 \
  --batchsize 32 \
  --lr 5e-3 \
  --num_epochs 300
```

**빠른 실험:**
```bash
python main_cl_fix.py \
  --model MILLET \
  --millet_pooling attention \
  --embed 64 \
  --batchsize 64 \
  --num_epochs 100
```

### 6.3 AOPCR 해석

**AOPCR 값 이해:**
- **AOPCR > 0**: 모델의 설명이 random보다 좋음 (바람직)
- **AOPCR ≈ 0**: 설명이 random과 비슷 (문제)
- **AOPCR < 0**: 설명이 random보다 나쁨 (심각한 문제)

**개선 방법:**
1. Conjunctive pooling 사용
2. Dropout 줄이기 (overfitting이 아니라면)
3. Positional encoding 활성화
4. Epoch 늘리기 (충분한 학습)

---

## 7. 참고 자료

### 7.1 관련 파일
- [models/milet.py](models/milet.py) - MILLET 모델 정의
- [models/inceptiontime.py](models/inceptiontime.py) - InceptionTime backbone
- [models/common.py](models/common.py) - 공통 유틸리티
- [main_cl_fix.py](main_cl_fix.py) - 학습/평가 파이프라인
- [compute_aopcr.py](compute_aopcr.py) - AOPCR 계산

### 7.2 원본 프로젝트
- **MILLET 저장소**: https://github.com/amazon-science/MILTimeSeriesClassification
- **InceptionTime 논문**: "InceptionTime: Finding AlexNet for time series classification" (Data Mining and Knowledge Discovery, 2020)

### 7.3 라이선스
- InceptionTime 코드: BSD 3-Clause (PyTorch 튜토리얼)
- MILLET 코드: Apache-2.0 (Amazon)

---

## 8. 변경 이력

| 날짜 | 버전 | 변경사항 |
|-----|------|----------|
| 2024-12-24 | 1.0 | ✅ MILLET 모델 초기 이식 완료 |
| | | - 5가지 pooling 방식 구현 |
| | | - main_cl_fix.py 통합 |
| | | - compute_aopcr.py MILLET 지원 |
| | | - DDP 호환성 확보 |
| 2024-12-24 | 1.1 | ✅ 출력 형식 및 문서 업데이트 |
| | | - is_instance=True시 7개 반환값으로 통일 (호환성) |
| | | - is_instance=False시 조건부 반환 (3개 또는 1개) |
| | | - max_seq_len 파라미터 추가 (호환성) |
| | | - 평가 로직 개선 (instance_pred 유지) |
| | | - 상세 문서화 최신화 |
| 2024-12-24 | 1.2 | ✅ 상세 수식 및 평가 메트릭 추가 |
| | | - 5가지 pooling 방식 수식 및 코드 상세 설명 |
| | | - Conjunctive vs Additive 비교표 |
| | | - MIL 가정과의 관계 설명 (Instance pooling) |
| | | - CAM (Class Activation Map) 설명 (GAP) |
| | | - 평가 메트릭 섹션 추가 (2.5) |
| | | - 분류 메트릭: Accuracy, Balanced Acc, AUROC, mAP, F1 |
| | | - 해석 메트릭: AOPCR, NDCG@N |
| | | - 메트릭별 수식, 코드, 사용 가이드 |

---

## 9. 추가 개발 계획

### 단기 (완료)
- [x] 기본 MILLET 이식
- [x] 5가지 pooling 방식
- [x] AOPCR 지원
- [x] DDP 지원

### 중기 (검토 중)
- [ ] Contrastive learning 통합 (TimeMIL의 prototype loss)
- [ ] Variable-length bag 네이티브 지원
- [ ] Visualization 도구 (attention, interpretation)
- [ ] 추가 pooling 방식 (gated, hierarchical)

### 장기
- [ ] Efficient attention 메커니즘 추가
- [ ] Hybrid model (Transformer + CNN)
- [ ] AutoML 하이퍼파라미터 튜닝

---

**문서 작성일**: 2024-12-24
**최종 수정일**: 2024-12-24
**MILLET 이식 버전**: 1.2
**문서 페이지 수**: ~40페이지 (A4 기준)
**원본 프로젝트**: [MILTimeSeriesClassification](https://github.com/amazon-science/MILTimeSeriesClassification)
