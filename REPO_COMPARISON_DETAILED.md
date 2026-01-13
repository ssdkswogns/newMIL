# 코드 구현 단위 상세 Repo 비교 문서

## 개요

이 문서는 두 repository의 코드 구현 수준에서의 차이점을 상세하게 분석합니다.

- **현재 Repo**: `/home/moon/code/newMIL` (최신 버전)
- **비교 Repo**: `/home/moon/code/newnewmil/newMIL` (구버전)

---

## 1. 메인 학습 파일 비교

### 1.1 main_cl_fix_ambiguous.py vs main_cl_fix.py

#### 파일 크기
- **현재**: 1,467 줄
- **비교**: 1,393 줄
- **차이**: +74 줄

#### A. Reproducibility 구현 차이 (Lines 54-83)

**현재 Repo - 강화된 재현성 함수:**
```python
def seed_everything(seed: int, deterministic: bool = True):
    """
    Seed python, numpy, torch (CPU/GPU) and toggle deterministic options.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # TF32 비활성화로 Ampere+ GPU에서 완전한 재현성 보장
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def seed_worker(worker_id: int):
    """DataLoader worker별 RNG seeding 보장"""
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**비교 Repo - 기본 seeding만 제공:**
```python
# Lines 75-85
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**주요 차이점:**
1. ✅ 현재: `PYTHONHASHSEED` 환경 변수 설정
2. ✅ 현재: TF32 명시적 비활성화 (A100/A6000 GPU 대응)
3. ✅ 현재: `torch.use_deterministic_algorithms(True)` 적용
4. ✅ 현재: DataLoader worker seeding 함수 제공
5. ❌ 비교: worker seeding 없음 → 멀티프로세스 데이터 로딩 시 재현성 문제

---

#### B. Model Import 차이 (Lines 38-44)

**현재 Repo - 5개 Baseline 모델 추가:**
```python
from models.timemil import TimeMIL, newTimeMIL, AmbiguousMIL, AmbiguousMILwithCL
from models.milet import MILLET
from models.MLSTM_FCN import MLSTM_FCN
from models.os_cnn import OS_CNN
from models.ED_1NN import ED1NN
from models.DTW_1NN_torch import dtw_1nn_D_predict_torch, dtw_1nn_I_predict_torch
from compute_aopcr import compute_classwise_aopcr
```

**비교 Repo - Experimental 모델:**
```python
from models.timemil import TimeMIL, newTimeMIL, AmbiguousMIL, AmbiguousMILwithCL
from models.expmil import AmbiguousMILwithCL as HybridMIL, ContextualAmbiguousMILwithCL as ContextualMIL
from compute_aopcr import compute_classwise_aopcr
```

**차이점:**
| 현재 Repo | 비교 Repo |
|-----------|-----------|
| MILLET (MIL + InceptionTime) | HybridMIL (Dual Pooling) |
| MLSTM_FCN (LSTM + CNN) | ContextualMIL (Self-Attention) |
| OS_CNN (Omni-Scale CNN) | - |
| ED_1NN (Euclidean Distance) | - |
| DTW_1NN_D/I (Dynamic Time Warping) | - |

---

#### C. train() 함수 - Forward Pass 로직 (Lines 443-470)

**현재 Repo - 다양한 모델 지원:**
```python
# MILLET 모델
if args.model == 'MILLET':
    bag_prediction, instance_pred, interpretation = milnet(bag_feats)
    x_cls, x_seq = None, None

# OS_CNN 모델 (CAM 지원)
elif args.model == 'OS_CNN':
    out = milnet(bag_feats, return_cam=True)
    bag_prediction, interpretation = out
    instance_pred = None
    x_cls, x_seq = None, None

# MLSTM_FCN 모델
elif args.model == 'MLSTM_FCN':
    bag_prediction = milnet(bag_feats)
    instance_pred = None
    x_cls, x_seq = None, None

# AmbiguousMIL (warmup 지원)
elif args.model == 'AmbiguousMIL':
    if epoch < args.epoch_des:
        bag_prediction, instance_pred, x_cls, x_seq, c_seq, attn_layer1, attn_layer2 = milnet(
            bag_feats, warmup=True
        )
    else:
        bag_prediction, instance_pred, x_cls, x_seq, c_seq, attn_layer1, attn_layer2 = milnet(
            bag_feats, warmup=False
        )
```

**비교 Repo - HybridMIL/ContextualMIL 지원:**
```python
elif args.model == 'HybridMIL' or args.model == 'ContextualMIL':
    # 8개의 출력값 반환
    outputs = milnet(bag_feats, warmup=(epoch < args.epoch_des))
    bag_logits_ts, bag_logits_tp, weighted_logits, instance_pred, x_cls, x_seq, attn_ts, attn_tp = outputs

    # bag_logits_tp를 메인 예측으로 사용
    bag_prediction = bag_logits_tp
    c_seq = None
    attn_layer1 = attn_ts
    attn_layer2 = attn_tp
```

**주요 차이:**
- **현재**: 각 baseline 모델의 고유한 출력 형식 처리
- **비교**: Dual pooling (TS: Time-Series attention, TP: Temporal pooling) 전략

---

#### D. Loss 계산 차이 (Lines 525-545)

**현재 Repo:**
```python
if args.model == 'AmbiguousMIL':
    loss = (
        args.bag_loss_w * bag_loss
        + args.inst_loss_w * inst_loss
        + args.proto_loss_w * proto_loss
        + args.smooth_loss_w * smooth_loss
    )
else:
    # TimeMIL, newTimeMIL 등 기본 모델
    loss = args.bag_loss_w * bag_loss + args.inst_loss_w * inst_loss
```

**비교 Repo:**
```python
if args.model == 'AmbiguousMIL' or args.model == 'HybridMIL' or args.model == 'ContextualMIL':
    loss = (
        args.bag_loss_w * bag_loss
        + args.inst_loss_w * inst_loss
        + args.proto_loss_w * proto_loss
        + args.smooth_loss_w * smooth_loss
    )
else:
    loss = args.bag_loss_w * bag_loss + args.inst_loss_w * inst_loss
```

**차이점:**
- 현재: AmbiguousMIL만 4개 loss 조합 사용
- 비교: HybridMIL, ContextualMIL도 동일한 loss 조합 사용

---

#### E. test() 함수 - Model Output 처리 (Lines 615-640)

**현재 Repo - 다양한 출력 형식 처리:**
```python
if args.model == 'AmbiguousMIL':
    bag_prediction, instance_pred, x_cls, x_seq, c_seq, attn_layer1, attn_layer2 = model(bag_feats)

elif args.model == 'MILLET':
    bag_prediction, instance_pred, interpretation = model(bag_feats)
    attn_layer2 = None
    x_seq = None

elif args.model == 'OS_CNN':
    out = model(bag_feats, return_cam=True)
    bag_prediction, interpretation = out
    instance_pred = None
    attn_layer2 = None

elif args.model == 'MLSTM_FCN':
    out = model(bag_feats)
    bag_prediction = out if not isinstance(out, (tuple, list)) else out[0]
    instance_pred = None
    attn_layer2 = None

elif args.model == 'newTimeMIL':
    out = model(bag_feats)
    instance_pred = None
    attn_layer2 = out[3] if len(out) >= 4 else None
```

**비교 Repo:**
```python
if args.model == 'AmbiguousMIL':
    bag_prediction, instance_pred, x_cls, x_seq, c_seq, attn_layer1, attn_layer2 = model(bag_feats)

elif args.model == 'HybridMIL' or args.model == 'ContextualMIL':
    outputs = model(bag_feats)
    bag_logits_ts, bag_logits_tp, weighted_logits, instance_pred, x_cls, x_seq, attn_ts, attn_tp = outputs
    bag_prediction = bag_logits_tp
    c_seq = None
    attn_layer1 = attn_ts
    attn_layer2 = attn_tp
```

---

#### F. Instance Accuracy 계산 (Lines 709-730)

**현재 Repo:**
```python
if args.datatype == "mixed" and (y_inst is not None):
    y_inst = y_inst.to(device)
    y_inst_label = torch.argmax(y_inst, dim=2)  # [B, T]

    if args.model == 'AmbiguousMIL':
        pred_inst = torch.argmax(instance_pred, dim=2)  # [B, T]
    elif args.model == 'MILLET':
        pred_inst = torch.argmax(instance_pred, dim=2)  # [B, T]
    elif args.model == 'newTimeMIL' and attn_layer2 is not None:
        B, T, C = y_inst.shape
        # Attention에서 class 정보 추출
        attn_cls = attn_layer2[:,:,:C,C:]
        attn_mean = attn_cls.mean(dim=1)
        pred_inst = torch.argmax(attn_mean, dim=1)
    else:
        pred_inst = None
```

**비교 Repo:**
```python
if args.datatype == "mixed" and (y_inst is not None):
    y_inst = y_inst.to(device)
    y_inst_label = torch.argmax(y_inst, dim=2)

    if args.model == 'AmbiguousMIL' or args.model == 'HybridMIL' or args.model == 'ContextualMIL':
        pred_inst = torch.argmax(instance_pred, dim=2)
    elif args.model == 'newTimeMIL' and attn_layer2 is not None:
        # (동일한 로직)
```

---

#### G. 1-NN Baseline 구현 (Lines 1036-1083, 현재 Repo만)

**ED_1NN 실행:**
```python
if args.model == 'ED1NN':
    if args.datatype != 'original':
        raise ValueError(f"ED1NN baseline supports only datatype=original")

    clf = ED1NN(normalize=args.ed1nn_normalize)
    clf.fit(Xtr_np, y_train_np)
    y_pred = clf.predict(Xte_np)

    bag_acc = accuracy_score(y_test_np, y_pred)
    bag_f1 = f1_score(y_test_np, y_pred, average='macro', zero_division=0)

    print(f"[ED1NN Results] Bag Acc={bag_acc:.4f}, Bag F1={bag_f1:.4f}")
```

**DTW_1NN_I (Independent) 실행:**
```python
elif args.model == 'DTW1NN_I':
    y_pred = dtw_1nn_I_predict_torch(
        Xtr_np, y_train_np, Xte_np,
        device=device,
        window=args.dtw_window,
        max_train=args.dtw_max_train,
        max_test=args.dtw_max_test
    )
    # Per-channel DTW distance 합산
```

**DTW_1NN_D (Dependent) 실행:**
```python
else:  # DTW1NN_D
    y_pred = dtw_1nn_D_predict_torch(
        Xtr_np, y_train_np, Xte_np,
        device=device,
        window=args.dtw_window,
        max_train=args.dtw_max_train,
        max_test=args.dtw_max_test
    )
    # Multivariate DTW (공유 warping path)
```

**비교 Repo에는 없음** - 오직 MIL 계열 모델만 지원

---

#### H. Command-line Arguments 차이

**현재 Repo 추가 인자:**
```python
parser.add_argument('--model', default='TimeMIL', type=str,
                    help='Model (TimeMIL | newTimeMIL | AmbiguousMIL | MILLET | MLSTM_FCN | OS_CNN | ED1NN | DTW1NN_D | DTW1NN_I)')

# MILLET 관련
parser.add_argument('--millet_pooling', default='conjunctive', type=str,
                    help="Pooling for MILLET (conjunctive | attention | instance | additive | gap)")

# 1-NN 관련
parser.add_argument('--ed1nn_normalize', action='store_true',
                    help='Normalize ED1NN (z-score per channel)')
parser.add_argument('--dtw_window', type=int, default=None,
                    help='DTW Sakoe-Chiba window size (None=no constraint)')
parser.add_argument('--dtw_max_train', type=int, default=None,
                    help='Limit training samples for DTW (speed)')
parser.add_argument('--dtw_max_test', type=int, default=None,
                    help='Limit test samples for DTW (speed)')
```

**비교 Repo 추가 인자:**
```python
# ContextualMIL 관련
parser.add_argument('--attn_layers', type=int, default=1,
                    help='Number of self-attention layers')
parser.add_argument('--attn_heads', type=int, default=4,
                    help='Number of attention heads')
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='Attention dropout')
parser.add_argument('--use_pos_enc', type=int, default=1,
                    help='Use positional encoding (0/1)')
```

---

#### I. Model 초기화 차이 (Lines 1086-1150)

**현재 Repo - Baseline 모델들:**
```python
elif args.model == 'MILLET':
    base_model = MILLET(
        args.feats_size,
        mDim=args.embed,
        n_classes=num_classes,
        dropout=args.dropout_node,
        max_seq_len=seq_len,
        pooling=args.millet_pooling,  # 5가지 pooling 전략
        is_instance=True
    ).to(device)
    proto_bank = None  # CL 사용 안함

elif args.model == 'OS_CNN':
    base_model = OS_CNN(
        in_channels=args.feats_size,
        n_class=num_classes,
        layer_parameter_list=None,  # default 2-block architecture
        few_shot=False
    ).to(device)
    proto_bank = None

elif args.model == 'MLSTM_FCN':
    base_model = MLSTM_FCN(
        input_dim=args.feats_size,
        num_classes=num_classes,
        lstm_hidden=8,
        conv_filters=(128, 256, 128),
        kernel_sizes=(8, 5, 3),
        lstm_dropout=0.8,
        se_reduction=16  # Squeeze-Excitation reduction ratio
    ).to(device)
    proto_bank = None
```

**비교 Repo - Experimental 모델들:**
```python
elif args.model == 'HybridMIL':
    base_model = HybridMIL(
        in_features=args.feats_size,
        n_classes=num_classes,
        mDim=args.embed,
        dropout=args.dropout_node,
        is_instance=True
    ).to(device)
    proto_bank = PrototypeBank(
        num_classes=num_classes,
        dim=args.embed,
        device=device,
        momentum=0.9
    )

elif args.model == 'ContextualMIL':
    base_model = ContextualMIL(
        in_features=args.feats_size,
        n_classes=num_classes,
        mDim=args.embed,
        dropout=args.dropout_node,
        is_instance=True,
        attn_layers=getattr(args, 'attn_layers', 1),
        attn_heads=getattr(args, 'attn_heads', 4),
        attn_dropout=getattr(args, 'attn_dropout', 0.1),
        use_pos_enc=bool(getattr(args, 'use_pos_enc', True)),
    ).to(device)
    proto_bank = PrototypeBank(...)  # CL 사용
```

---

## 2. compute_aopcr.py 비교

### 2.1 함수 시그니처 차이

**현재 Repo (Lines 83-95):**
```python
def compute_classwise_aopcr(
    milnet,
    testloader,
    args,
    stop: float = 0.5,               # 기존
    step: float = 0.05,              # 기존
    n_random: int = 3,               # 기존
    pred_threshold: float = 0.5,     # 기존
    coverage_thresholds: list = None,  # ✅ 신규
    rng: torch.Generator = None,       # ✅ 신규
):
    """
    Coverage thresholds: [0.9, 0.8, 0.7, 0.5] 등
    → 원래 성능의 X%를 유지하는데 필요한 인스턴스 비율 계산
    """
```

**비교 Repo (Lines 83-91):**
```python
def compute_classwise_aopcr(
    milnet,
    testloader,
    args,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    pred_threshold: float = 0.5,
):
    # coverage 계산 없음
```

---

### 2.2 Coverage Metrics 구현 (현재 Repo만)

**초기화 (Lines 131-139):**
```python
if coverage_thresholds is None:
    coverage_thresholds = []

# 각 threshold별로 class당 coverage 저장
coverage_stats_expl = {
    thr: torch.zeros(num_classes, device=device)
    for thr in coverage_thresholds
}
coverage_stats_rand = {
    thr: torch.zeros(num_classes, device=device)
    for thr in coverage_thresholds
}
```

**계산 로직 (Lines 315-339):**
```python
# Perturbation curve에서 성능 threshold 도달 시점 찾기
for thr in coverage_thresholds:
    target_logit = orig_logit * thr  # 예: 0.9 * orig = 90% 성능

    # Explanation-based: 몇 %까지 제거해도 90% 성능 유지?
    mask_expl = curve_expl >= target_logit
    if mask_expl.any():
        last_valid_step = mask_expl.nonzero(as_tuple=False).max().item()
        coverage_expl = alphas[last_valid_step].item()
    else:
        coverage_expl = 0.0

    # Random-based: 평균적으로 몇 % 제거 가능?
    curve_rand_mean = curves_rand.mean(dim=0)
    mask_rand = curve_rand_mean >= target_logit
    if mask_rand.any():
        last_valid_step = mask_rand.nonzero(as_tuple=False).max().item()
        coverage_rand = alphas[last_valid_step].item()
    else:
        coverage_rand = 0.0

    coverage_stats_expl[thr][pred_c] += coverage_expl
    coverage_stats_rand[thr][pred_c] += coverage_rand
```

**출력 형식 (Lines 370-395):**
```python
coverage_summary = {}
for thr in coverage_thresholds:
    cov_expl_avg = (coverage_stats_expl[thr] / counts.clamp(min=1)).cpu().numpy()
    cov_rand_avg = (coverage_stats_rand[thr] / counts.clamp(min=1)).cpu().numpy()

    # Per-class weighted average
    total_samples = counts.sum().item()
    cov_expl_weighted = (coverage_stats_expl[thr].sum() / total_samples).item()
    cov_rand_weighted = (coverage_stats_rand[thr].sum() / total_samples).item()

    # Gain: explanation이 random보다 얼마나 더 좋은가
    gain_weighted = cov_expl_weighted - cov_rand_weighted

    coverage_summary[thr] = {
        "explanation": cov_expl_weighted,
        "random": cov_rand_weighted,
        "gain": gain_weighted,
        "per_class_expl": cov_expl_avg.tolist(),
        "per_class_rand": cov_rand_avg.tolist(),
    }
```

**비교 Repo: Coverage 계산 없음**

---

### 2.3 모델별 Forward Pass 차이

**현재 Repo (Lines 155-202):**
```python
# MILLET
elif args.model == 'MILLET':
    logits, non_weighted_instance_logit, instance_logits = milnet(x)
    prob = torch.sigmoid(logits)
    interpretation = None  # MILLET는 interpretation 별도 제공 안함

# MLSTM_FCN
elif args.model == 'MLSTM_FCN':
    out = milnet(x, return_cam=True)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, interpretation = out  # [B, C, T] CAM
    else:
        logits = out
        interpretation = None
    prob = torch.sigmoid(logits)

# OS_CNN
elif args.model == 'OS_CNN':
    out = milnet(x, return_cam=True)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, interpretation = out  # [B, C, T] CAM
    else:
        logits = out
        interpretation = None
    prob = torch.sigmoid(logits)
```

**비교 Repo (Lines 134-163):**
```python
# HybridMIL / ContextualMIL
elif args.model == 'HybridMIL' or args.model == 'ContextualMIL':
    out = milnet(x)
    # 8-tuple unpacking
    bag_logits_ts, bag_logits_tp, weighted_logits, inst_logits, \
        bag_emb, inst_emb, attn_ts, attn_tp = out

    logits = bag_logits_tp  # TP (Temporal Pooling) 사용
    prob = torch.sigmoid(logits)
    instance_logits = weighted_logits  # [B, T, C] for interpretation
```

---

### 2.4 Interpretation 방식 차이

**현재 Repo - 다양한 해석 방법:**
```python
# AmbiguousMIL: instance_logits 직접 사용
if args.model == 'AmbiguousMIL':
    if instance_logits is not None:
        interpretation = instance_logits.transpose(1, 2)  # [B, C, T]

# OS_CNN/MLSTM_FCN: CAM (Class Activation Map)
elif args.model in ['OS_CNN', 'MLSTM_FCN']:
    # interpretation은 forward에서 반환된 CAM 사용

# MILLET: instance prediction 기반
elif args.model == 'MILLET':
    if instance_logits is not None:
        interpretation = instance_logits.transpose(1, 2)

# TimeMIL/newTimeMIL: Attention weights 사용
else:
    if attn_layer2 is not None:
        attn_cls = attn_layer2[:, :, :C, C:]  # Class-related attention
        interpretation = attn_cls.mean(dim=1)  # [B, C, T]
```

**비교 Repo:**
```python
# HybridMIL/ContextualMIL: weighted_logits 사용
if args.model in ['HybridMIL', 'ContextualMIL']:
    interpretation = weighted_logits.transpose(1, 2)  # [B, C, T]
    # weighted_logits = instance_logits * attn_tp

# 기타 모델: Attention 기반
else:
    if attn_layer2 is not None:
        attn_cls = attn_layer2[:, :, :C, C:]
        interpretation = attn_cls.mean(dim=1)
```

---

### 2.5 Return 값 차이

**현재 Repo:**
```python
return (
    aopcr_per_class.cpu().numpy(),      # [num_classes]
    aopcr_weighted,                     # float
    aopcr_mean,                         # float
    aopcr_overall_mean,                 # float or None
    M_expl.cpu().numpy(),               # [num_steps, num_classes]
    M_rand.cpu().numpy(),               # [num_steps, num_classes]
    alphas.cpu().numpy(),               # [num_steps]
    counts.cpu().numpy(),               # [num_classes]
    coverage_summary,                   # ✅ dict with coverage metrics
)
```

**비교 Repo:**
```python
return (
    aopcr_per_class.cpu().numpy(),
    aopcr_weighted,
    aopcr_mean,
    aopcr_overall_mean,
    M_expl.cpu().numpy(),
    M_rand.cpu().numpy(),
    alphas.cpu().numpy(),
    counts.cpu().numpy(),
    # coverage_summary 없음
)
```

---

## 3. eval.py 비교

### 3.1 evaluate_classification() 함수

**현재 Repo - 확장된 모델 지원 (Lines 64-95):**
```python
# AmbiguousMIL
if args.model == 'AmbiguousMIL':
    if not isinstance(out, (tuple, list)) or len(out) < 6:
        raise ValueError("Unexpected AmbiguousMIL output")
    logits, instance_pred = out[0], out[1]
    attn_layer2 = out[-1]

# MILLET
elif args.model == 'MILLET':
    if not isinstance(out, (tuple, list)) or len(out) < 3:
        raise ValueError("Unexpected MILLET output")
    logits, non_weighted_instance_pred, instance_pred = out
    attn_layer2 = None

# OS_CNN
elif args.model == 'OS_CNN':
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, cam = out
    else:
        logits = out
    instance_pred = None
    attn_layer2 = None

# MLSTM_FCN
elif args.model == 'MLSTM_FCN':
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, cam = out
    else:
        logits = out
    instance_pred = None
    attn_layer2 = None

# TimeMIL / newTimeMIL
elif args.model in ['TimeMIL', 'newTimeMIL']:
    if not isinstance(out, (tuple, list)) or len(out) < 4:
        raise ValueError("Unexpected model output")
    logits, _, _, attn_layer2 = out
    instance_pred = None
```

**비교 Repo (Lines 64-76):**
```python
if args.model == 'AmbiguousMIL':
    if not isinstance(out, (tuple, list)) or len(out) < 6:
        raise ValueError("Unexpected AmbiguousMIL output")
    logits, instance_pred, pred_inst = out[0], out[1], out[2]
    attn_layer2 = None

else:
    # 모든 다른 모델
    if not isinstance(out, (tuple, list)) or len(out) < 4:
        raise ValueError("Unexpected model output")
    logits, _, _, attn_layer2 = out
    attn_cls = attn_layer2[:, :, :C, C:]
    attn_mean = attn_cls.mean(dim=1)
    instance_pred = None
```

---

### 3.2 AOPCR 실행 차이

**현재 Repo - Configurable Parameters (Lines 180-195):**
```python
# Command-line arguments
parser.add_argument('--aopcr_stop', default=0.5, type=float,
                    help='AOPCR perturbation limit (0.0-1.0). Default 0.5 (50%). Use 1.0 for full range.')
parser.add_argument('--aopcr_step', default=0.05, type=float,
                    help='AOPCR perturbation step size. Default 0.05 (5%).')

# Execution
result = compute_classwise_aopcr(
    milnet,
    testloader,
    args,
    stop=args.aopcr_stop,          # ✅ Configurable
    step=args.aopcr_step,          # ✅ Configurable
    n_random=args.num_random,
    pred_threshold=0.5,
    coverage_thresholds=[0.9, 0.8, 0.7, 0.5]  # ✅ Coverage metrics
)

# Unpacking with coverage
aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_sum, M_expl, M_rand, alphas, counts, coverage_summary = result
```

**비교 Repo - Fixed Parameters:**
```python
# No command-line arguments for stop/step

aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_sum, M_expl, M_rand, alphas, counts = compute_classwise_aopcr(
    milnet,
    testloader,
    args,
    stop=0.5,           # ❌ Hardcoded
    step=0.05,          # ❌ Hardcoded
    n_random=args.num_random,
    pred_threshold=0.5
)
# No coverage_summary
```

---

### 3.3 Coverage 출력 (현재 Repo만)

**Lines 220-235:**
```python
if coverage_summary:
    print("\n=== Coverage Analysis ===")
    for thr in sorted(coverage_summary.keys(), reverse=True):
        info = coverage_summary[thr]
        print(f"\nCoverage@{int(thr*100)}%:")
        print(f"  Explanation: {info['explanation']:.4f} (mean: {np.mean(info['per_class_expl']):.4f})")
        print(f"  Random:      {info['random']:.4f} (mean: {np.mean(info['per_class_rand']):.4f})")
        print(f"  Gain:        {info['gain']:.4f}")

# 예시 출력:
# Coverage@90%:
#   Explanation: 0.3245 (mean: 0.3156)
#   Random:      0.5678 (mean: 0.5542)
#   Gain:        -0.2433
#
# → 90% 성능 유지를 위해 explanation 기반은 32.45%만 남기면 됨
#   (random은 56.78% 필요 → explanation이 더 효율적)
```

**비교 Repo: 없음**

---

## 4. 새로운 Baseline 모델 구현

### 4.1 MILLET (models/milet.py)

**파일 크기:** 약 550줄

#### 핵심 아키텍처:

**1. InceptionTime Feature Extractor:**
```python
class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_size=40, bottleneck_channels=32, n_blocks=1):
        # InceptionBlock 여러 개 쌓기
        self.blocks = nn.ModuleList([
            InceptionBlock(in_channels, n_filters, bottleneck_channels, kernel_size)
            for _ in range(n_blocks)
        ])
        self.residual_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, bottleneck_channels, kernel_size):
        # 3개의 다른 kernel size로 병렬 convolution
        self.conv1 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_size // 4)
        self.conv2 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_size // 2)
        self.conv3 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
```

**2. 5가지 Pooling 전략:**

```python
# (1) Conjunctive Pooling (Default)
class MILConjunctivePooling(MILPooling):
    """Instance attention으로 weighted average"""
    def forward(self, instance_embeddings):
        attn = self.attention_head(instance_embeddings)           # [B, T, 1]
        instance_logits = self.instance_classifier(instance_embeddings)  # [B, T, C]
        weighted = instance_logits * attn
        bag_logits = torch.mean(weighted, dim=1)
        return {"bag_logits": bag_logits, "interpretation": weighted.transpose(1, 2)}

# (2) Attention Pooling
class MILAttentionPooling(MILPooling):
    """Class-wise attention pooling (Ilse et al.)"""
    def forward(self, instance_embeddings):
        attn = self.attention_head(instance_embeddings)           # [B, T, C]
        attn_weights = F.softmax(attn.transpose(1, 2), dim=-1)    # [B, C, T]
        bag_embedding = torch.bmm(attn_weights, instance_embeddings)  # [B, C, D]
        bag_logits = self.bag_classifier(bag_embedding).squeeze(-1)   # [B, C]
        return {"bag_logits": bag_logits}

# (3) Instance Pooling
class MILInstancePooling(MILPooling):
    """Max pooling of instance predictions"""
    def forward(self, instance_embeddings):
        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits, _ = torch.max(instance_logits, dim=1)
        return {"bag_logits": bag_logits}

# (4) Additive Pooling
class MILAdditivePooling(MILPooling):
    """Element-wise max + avg pooling"""
    def forward(self, instance_embeddings):
        max_pool, _ = torch.max(instance_embeddings, dim=1)
        avg_pool = torch.mean(instance_embeddings, dim=1)
        combined = max_pool + avg_pool
        bag_logits = self.bag_classifier(combined)
        return {"bag_logits": bag_logits}

# (5) GAP (Global Average Pooling)
class MILGAPPooling(MILPooling):
    """Simple average pooling"""
    def forward(self, instance_embeddings):
        avg = torch.mean(instance_embeddings, dim=1)
        bag_logits = self.bag_classifier(avg)
        return {"bag_logits": bag_logits}
```

**3. MILLET Main Model:**
```python
class MILLET(nn.Module):
    def __init__(
        self,
        feats_size: int,
        mDim: int = 128,
        n_classes: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pooling: str = "conjunctive",
        is_instance: bool = False,
        apply_positional_encoding: bool = True,
        padding_mode: str = "replicate",
    ):
        # Feature extractor
        self.feature_extractor = InceptionTimeFeatureExtractor(
            in_channels=feats_size,
            n_filters=mDim,
            kernel_size=40,
            bottleneck_channels=32,
            n_blocks=1
        )

        # Positional encoding
        if apply_positional_encoding:
            self.pos_enc = SinusoidalPositionalEncoding(mDim)

        # Pooling layer
        if pooling == "conjunctive":
            self.pooling = MILConjunctivePooling(mDim, n_classes, dropout, is_instance)
        elif pooling == "attention":
            self.pooling = MILAttentionPooling(mDim, n_classes, dropout)
        # ... 다른 pooling 옵션들

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        inst_emb = self.feature_extractor(x)  # [B, mDim, T]
        inst_emb = inst_emb.transpose(1, 2)   # [B, T, mDim]

        if hasattr(self, 'pos_enc'):
            inst_emb = self.pos_enc(inst_emb)

        pooling_output = self.pooling(inst_emb)
        bag_logits = pooling_output["bag_logits"]

        # Return format for compatibility
        if self.is_instance:
            instance_logits = pooling_output.get("instance_logits")
            interpretation = pooling_output.get("interpretation")
            return bag_logits, instance_logits, interpretation
        else:
            return bag_logits
```

---

### 4.2 MLSTM_FCN (models/MLSTM_FCN.py)

**파일 크기:** 약 180줄

#### 아키텍처:

**1. Squeeze-and-Excitation Block:**
```python
class SqueezeExcite(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        squeeze = F.adaptive_avg_pool1d(x, 1).view(B, C)  # [B, C]
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))  # [B, C]
        return x * excitation.view(B, C, 1)
```

**2. MLSTM_FCN Main Model:**
```python
class MLSTM_FCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_hidden: int = 8,
        conv_filters: Tuple[int, int, int] = (128, 256, 128),
        kernel_sizes: Tuple[int, int, int] = (8, 5, 3),
        lstm_dropout: float = 0.8,
        se_reduction: int = 16,
    ):
        # ===== LSTM Branch =====
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # ===== FCN Branch with SE =====
        f1, f2, f3 = conv_filters
        k1, k2, k3 = kernel_sizes

        # Conv Block 1
        self.conv1 = nn.Conv1d(input_dim, f1, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(f1)
        self.se1 = SqueezeExcite(f1, se_reduction)

        # Conv Block 2
        self.conv2 = nn.Conv1d(f1, f2, kernel_size=k2, padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(f2)
        self.se2 = SqueezeExcite(f2, se_reduction)

        # Conv Block 3
        self.conv3 = nn.Conv1d(f2, f3, kernel_size=k3, padding=k3 // 2)
        self.bn3 = nn.BatchNorm1d(f3)
        self.se3 = SqueezeExcite(f3, se_reduction)

        # Final classifier (combine LSTM + FCN)
        self.classifier = nn.Linear(lstm_hidden + f3, num_classes)

    def forward(self, x, return_cam=False):
        # x: [B, T, C]
        B, T, C = x.shape

        # ===== LSTM Branch =====
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, lstm_hidden]
        lstm_features = self.lstm_dropout(lstm_out[:, -1, :])  # 마지막 timestep [B, lstm_hidden]

        # ===== FCN Branch =====
        x_conv = x.transpose(1, 2)  # [B, C, T]

        # Block 1
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.se1(x_conv)

        # Block 2
        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.se2(x_conv)

        # Block 3
        x_conv = self.conv3(x_conv)
        x_conv = self.bn3(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.se3(x_conv)

        # Global Average Pooling
        fcn_features = F.adaptive_avg_pool1d(x_conv, 1).squeeze(-1)  # [B, f3]

        # Concatenate LSTM + FCN
        combined = torch.cat([lstm_features, fcn_features], dim=1)  # [B, lstm_hidden + f3]
        logits = self.classifier(combined)  # [B, num_classes]

        if return_cam:
            # CAM: use last conv features
            cam = x_conv  # [B, f3, T]
            # Weight by classifier weights
            weights = self.classifier.weight[:, lstm_hidden:]  # [num_classes, f3]
            cam = torch.einsum('nct,mc->nmt', cam, weights)  # [B, num_classes, T]
            return logits, cam
        else:
            return logits
```

---

### 4.3 OS_CNN (models/os_cnn.py)

**파일 크기:** 약 280줄

#### 핵심 개념: Omni-Scale Convolution

**1. Layer Mask Creation:**
```python
def creak_layer_mask(layer_parameter_list: List[Tuple[int, int, int]]):
    """
    layer_parameter_list: [(in_ch, out_ch, kernel_size), ...]

    예: [(1, 32, 7), (1, 64, 5), (1, 32, 3)]
    → 3개의 다른 kernel size를 하나의 masked convolution으로 합침
    """
    # 가장 큰 kernel size 찾기
    max_kernel = max([param[2] for param in layer_parameter_list])
    max_in_ch = max([param[0] for param in layer_parameter_list])
    max_out_ch = sum([param[1] for param in layer_parameter_list])

    # Initialize weights and masks
    os_mask = torch.zeros(max_out_ch, max_in_ch, max_kernel)
    init_weight = torch.zeros(max_out_ch, max_in_ch, max_kernel)
    init_bias = torch.zeros(max_out_ch)

    out_ch_offset = 0
    for in_ch, out_ch, kernel_size in layer_parameter_list:
        center = max_kernel // 2
        kernel_offset = kernel_size // 2

        # Mask: kernel 위치만 1로 설정
        os_mask[
            out_ch_offset : out_ch_offset + out_ch,
            :in_ch,
            center - kernel_offset : center + kernel_offset + 1
        ] = 1

        # Weights: He initialization
        init_weight[
            out_ch_offset : out_ch_offset + out_ch,
            :in_ch,
            center - kernel_offset : center + kernel_offset + 1
        ] = torch.nn.init.kaiming_uniform_(
            torch.empty(out_ch, in_ch, kernel_size), a=math.sqrt(5)
        )

        out_ch_offset += out_ch

    return os_mask, init_weight, init_bias
```

**2. OS Block:**
```python
class OSBlock(nn.Module):
    def __init__(
        self,
        layer_parameters: List[Tuple[int, int, int]],
        squeeze_ratio: int = 4,
    ):
        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        # Masked convolution
        self.weight = nn.Parameter(init_weight)
        self.bias = nn.Parameter(init_bias)
        self.register_buffer('mask', os_mask)

        out_channels = init_weight.shape[0]

        # Squeeze-and-Excitation
        self.fc1 = nn.Linear(out_channels, out_channels // squeeze_ratio)
        self.fc2 = nn.Linear(out_channels // squeeze_ratio, out_channels)

    def forward(self, x):
        # Masked convolution
        masked_weight = self.weight * self.mask
        x = F.conv1d(x, masked_weight, bias=self.bias, padding='same')
        x = F.relu(x)

        # Squeeze-and-Excitation
        B, C, T = x.shape
        squeeze = F.adaptive_avg_pool1d(x, 1).view(B, C)
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        x = x * excitation.view(B, C, 1)

        return x
```

**3. OS_CNN Main Model:**
```python
class OS_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_class: int,
        layer_parameter_list: Optional[List[List[Tuple[int, int, int]]]] = None,
        few_shot: bool = False,
    ):
        if layer_parameter_list is None:
            # Default architecture: 2 OS blocks
            layer_parameter_list = [
                # Block 1: 3 scales
                [(in_channels, 32, 7), (in_channels, 64, 5), (in_channels, 32, 3)],
                # Block 2: 2 scales
                [(128, 64, 5), (128, 64, 3)],
            ]

        self.blocks = nn.ModuleList([
            OSBlock(params) for params in layer_parameter_list
        ])

        # Calculate output channels
        final_out_ch = sum([p[1] for p in layer_parameter_list[-1]])

        # Classifier
        if few_shot:
            self.classifier = None  # Metric learning 사용
        else:
            self.classifier = nn.Linear(final_out_ch, n_class)

    def forward(self, x, return_cam=False):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)

        # OS blocks
        for block in self.blocks:
            x = block(x)  # [B, final_out_ch, T]

        # Global Average Pooling
        features = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, final_out_ch]

        if self.classifier is not None:
            logits = self.classifier(features)  # [B, n_class]
        else:
            logits = features  # Few-shot: return features

        if return_cam:
            # CAM from last conv features
            if self.classifier is not None:
                weights = self.classifier.weight  # [n_class, final_out_ch]
                cam = torch.einsum('nct,mc->nmt', x, weights)  # [B, n_class, T]
            else:
                cam = x
            return logits, cam
        else:
            return logits
```

**Default Architecture:**
- Block 1: 3 parallel convolutions (kernel 7, 5, 3) → 128 channels
- Block 2: 2 parallel convolutions (kernel 5, 3) → 128 channels
- Each block has SE (Squeeze-and-Excitation)
- Global Average Pooling + Linear classifier

---

### 4.4 ED_1NN (models/ED_1NN.py)

**파일 크기:** 약 60줄

#### 구현:

```python
class ED1NN:
    """Euclidean Distance 1-Nearest Neighbor"""

    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: z-score normalization per channel
        """
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.mean_ = None
        self.std_ = None

    def _transform(self, X, fit=False):
        """Apply z-score normalization"""
        X = X.reshape(X.shape[0], -1).astype(np.float64)  # [N, T*C]

        if self.normalize:
            if fit:
                self.mean_ = X.mean(axis=0, keepdims=True)
                self.std_ = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - self.mean_) / self.std_

        return X

    def fit(self, X, y):
        """Store training data"""
        self.X_train = self._transform(X, fit=True)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        1-NN prediction using Euclidean distance.

        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a · b)
        """
        X_proc = self._transform(X, fit=False)  # [N_test, D]

        # Squared norms
        X2 = np.sum(X_proc * X_proc, axis=1, keepdims=True)  # [N_test, 1]
        T2 = np.sum(self.X_train * self.X_train, axis=1)[None]  # [1, N_train]

        # Distance matrix
        d2 = X2 + T2 - 2.0 * (X_proc @ self.X_train.T)  # [N_test, N_train]

        # Nearest neighbor
        nn_idx = np.argmin(d2, axis=1)

        return self.y_train[nn_idx]
```

**특징:**
- Scikit-learn 스타일 API (fit/predict)
- Optional z-score normalization
- Efficient distance computation using matrix operations
- No hyperparameters

---

### 4.5 DTW_1NN (models/DTW_1NN_torch.py)

**파일 크기:** 약 250줄

#### 두 가지 DTW 구현:

**1. DTW_1NN_I (Independent):**
```python
@torch.no_grad()
def dtw_1nn_I_predict_torch(
    X_train, y_train, X_test,
    device='cuda',
    window=None,
    max_train=None,
    max_test=None
):
    """
    Independent DTW: 각 채널별로 DTW 거리 계산 후 합산

    Args:
        X_train: [N_train, T_train, C]
        y_train: [N_train]
        X_test: [N_test, T_test, C]
        window: Sakoe-Chiba window size (None = no constraint)
        max_train: Limit training samples (for speed)
        max_test: Limit test samples (for speed)
    """
    Xtr = torch.from_numpy(X_train).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    if max_train:
        Xtr = Xtr[:max_train]
        y_train = y_train[:max_train]
    if max_test:
        Xte = Xte[:max_test]

    y_pred = []

    for x in Xte:  # x: [T_test, C]
        best_dist = float('inf')
        best_label = -1

        for xt, yt in zip(Xtr, y_train):  # xt: [T_train, C]
            # Sum DTW distance across channels
            d = 0.0
            for c in range(x.shape[1]):
                d = d + _dtw_distance_1d(
                    x[:, c], xt[:, c], window=window
                ).item()

            if d < best_dist:
                best_dist = d
                best_label = yt

        y_pred.append(best_label)

    return np.array(y_pred)

def _dtw_distance_1d(x, y, window=None):
    """
    1D DTW using dynamic programming.

    D[i, j] = cost[i, j] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    """
    n, m = len(x), len(y)
    D = torch.full((n + 1, m + 1), float('inf'), device=x.device)
    D[0, 0] = 0

    for i in range(1, n + 1):
        if window is None:
            j_start, j_end = 1, m + 1
        else:
            # Sakoe-Chiba band
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + torch.min(torch.stack([
                D[i - 1, j],      # insertion
                D[i, j - 1],      # deletion
                D[i - 1, j - 1]   # match
            ]))

    return torch.sqrt(D[n, m])
```

**2. DTW_1NN_D (Dependent):**
```python
@torch.no_grad()
def dtw_1nn_D_predict_torch(
    X_train, y_train, X_test,
    device='cuda',
    window=None,
    max_train=None,
    max_test=None
):
    """
    Dependent DTW: Multivariate DTW with shared warping path

    모든 채널이 동일한 warping path를 공유
    """
    Xtr = torch.from_numpy(X_train).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    if max_train:
        Xtr = Xtr[:max_train]
        y_train = y_train[:max_train]
    if max_test:
        Xte = Xte[:max_test]

    y_pred = []

    for x in Xte:  # x: [T_test, C]
        best_dist = float('inf')
        best_label = -1

        for xt, yt in zip(Xtr, y_train):  # xt: [T_train, C]
            # Multivariate DTW
            d = _dtw_distance_dependent(x, xt, window=window).item()

            if d < best_dist:
                best_dist = d
                best_label = yt

        y_pred.append(best_label)

    return np.array(y_pred)

def _dtw_distance_dependent(x, y, window=None):
    """
    Multivariate DTW: shared warping path across channels

    D[i, j] = ||x[i] - y[j]||^2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    """
    n, m = x.shape[0], y.shape[0]
    D = torch.full((n + 1, m + 1), float('inf'), device=x.device)
    D[0, 0] = 0

    for i in range(1, n + 1):
        if window is None:
            j_start, j_end = 1, m + 1
        else:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            # Euclidean distance across all channels
            cost = torch.sum((x[i - 1] - y[j - 1]) ** 2)
            D[i, j] = cost + torch.min(torch.stack([
                D[i - 1, j],
                D[i, j - 1],
                D[i - 1, j - 1]
            ]))

    return torch.sqrt(D[n, m])
```

**차이점:**
- **DTW_1NN_I**: 각 채널 독립적으로 DTW → 합산 (더 유연)
- **DTW_1NN_D**: 모든 채널이 동일한 warping 공유 (더 제약적)

---

## 5. 비교 Repo 전용 모델

### 5.1 HybridMIL (models/expmil.py)

**핵심 아이디어: Dual Pooling**

```python
class AmbiguousMILwithCL(nn.Module):
    """
    HybridMIL: 두 가지 pooling 전략 동시 사용

    1. TS (Time-Series): Attention pooling
    2. TP (Temporal): Conjunctive pooling
    """

    def __init__(self, in_features, n_classes, mDim=128, dropout=0.0, is_instance=True):
        # Feature extractor
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv1d(in_features, mDim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(mDim // 4, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== TS Head (Attention Pooling) =====
        self.attn_head_ts = nn.Sequential(
            nn.Linear(mDim, 8 * n_classes),
            nn.Tanh(),
            nn.Linear(8 * n_classes, n_classes),
            nn.Sigmoid(),
        )
        self.bag_classifier_ts = nn.Linear(mDim, 1)

        # ===== TP Head (Conjunctive Pooling) =====
        self.attn_head_tp = nn.Sequential(
            nn.Linear(mDim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.inst_classifier = nn.Linear(mDim, n_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]

        # Feature extraction
        feats = self.feature_extractor_part1(x)  # [B, mDim//4, 1]
        feats = feats.squeeze(2)  # [B, mDim//4]
        feats = feats.unsqueeze(1).expand(-1, x.shape[2], -1)  # [B, T, mDim//4]
        inst_emb = self.feature_extractor_part2(feats)  # [B, T, mDim]

        B, T, mDim = inst_emb.shape

        # ===== TS Prediction (Attention Pooling) =====
        score_ts = self.attn_head_ts(inst_emb)              # [B, T, C]
        attn_ts = F.softmax(score_ts.permute(0, 2, 1), dim=-1)  # [B, C, T]
        bag_emb = torch.bmm(attn_ts, inst_emb)              # [B, C, mDim]
        bag_emb_flat = bag_emb.view(B, -1)                  # [B, C * mDim]
        bag_logits_ts = self.bag_classifier_ts(bag_emb_flat).view(B, n_classes)

        # ===== TP Prediction (Conjunctive Pooling) =====
        attn_tp = self.attn_head_tp(inst_emb)               # [B, T, 1]
        inst_logits = self.inst_classifier(inst_emb)        # [B, T, C]
        weighted_logits = inst_logits * attn_tp             # [B, T, C]
        bag_logits_tp = weighted_logits.mean(dim=1)         # [B, C]

        return (
            bag_logits_ts,    # TS prediction
            bag_logits_tp,    # TP prediction (main)
            weighted_logits,  # For interpretation
            inst_logits,      # Instance predictions
            bag_emb,          # Bag embeddings (for CL)
            inst_emb,         # Instance embeddings (for CL)
            attn_ts,          # TS attention weights
            attn_tp,          # TP attention weights
        )
```

**사용 예:**
```python
# Training
bag_logits_ts, bag_logits_tp, weighted_logits, inst_logits, bag_emb, inst_emb, attn_ts, attn_tp = model(x)

# bag_logits_tp를 메인 예측으로 사용
bag_loss = BCE(bag_logits_tp, bag_label)

# Instance loss
inst_loss = BCE(weighted_logits, inst_label)

# Prototypical CL on bag_emb
proto_loss = compute_proto_loss(bag_emb, bag_label, proto_bank)
```

---

### 5.2 ContextualMIL (models/expmil.py)

**HybridMIL + Self-Attention Context**

```python
class ContextualAmbiguousMILwithCL(nn.Module):
    """
    ContextualMIL: HybridMIL + Multi-head Self-Attention

    Instance embedding에 contextual information 주입
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        mDim: int = 128,
        dropout: float = 0.0,
        is_instance: bool = True,
        attn_layers: int = 1,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_pos_enc: bool = True,
    ):
        # (HybridMIL과 동일한 구조)
        self.feature_extractor_part1 = ...
        self.feature_extractor_part2 = ...

        # ===== Contextual Layers =====
        if use_pos_enc:
            self.pos_enc = SinusoidalPositionalEncoding(mDim)
        else:
            self.pos_enc = None

        self.context_blocks = nn.ModuleList([
            _SelfAttnBlock(dim=mDim, heads=attn_heads, dropout=attn_dropout)
            for _ in range(attn_layers)
        ])

        # TS/TP heads (HybridMIL과 동일)
        self.attn_head_ts = ...
        self.bag_classifier_ts = ...
        self.attn_head_tp = ...
        self.inst_classifier = ...

    def forward(self, x):
        # Feature extraction
        inst_emb = self.feature_extractor_part2(...)  # [B, T, mDim]

        # ===== Inject Context =====
        if self.pos_enc is not None:
            inst_emb = self.pos_enc(inst_emb)

        for block in self.context_blocks:
            inst_emb = block(inst_emb)  # Self-attention

        # (이후 HybridMIL과 동일)
        # TS prediction
        # TP prediction

        return (bag_logits_ts, bag_logits_tp, weighted_logits, inst_logits,
                bag_emb, inst_emb, attn_ts, attn_tp)


class _SelfAttnBlock(nn.Module):
    """Self-attention block with residual connection"""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Feed-forward
        x = x + self.ffn(self.norm2(x))

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Transformer-style positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:x.size(1)]
```

**ContextualMIL 특징:**
- 각 instance가 다른 instance들의 정보를 참고 (self-attention)
- Positional encoding으로 순서 정보 보존
- Transformer 스타일의 LayerNorm + Residual connection

---

## 6. 유틸리티 코드 차이

### 6.1 collect_aopcr.py (현재 Repo만)

**목적:** Train_log.log 파일에서 AOPCR 결과 자동 추출

**주요 기능:**
```python
def parse_log(log_path: Path) -> Dict[str, Optional[Dict]]:
    """
    Parse Train_log.log and extract:
    1. Best Results (Bag Acc, Bag F1, Inst Acc, Inst F1)
    2. AOPCR per class
    3. AOPCR summary (Weighted, Mean, Sum)
    4. Coverage metrics (@90%, @80%, etc.)
    """

    # Regex patterns
    SUMMARY_RE = re.compile(r"Weighted AOPCR:\s*([0-9eE+\-\.]+),\s*Mean AOPCR:\s*([0-9eE+\-\.]+)")
    CLASS_RE = re.compile(r"\[AOPCR\]\s*class\s+(\S+)\s+count=(\d+)\s+AOPCR=([0-9eE+\-\.]+|NaN)")
    COVERAGE_RE = re.compile(r"Coverage@(\d+)%:")

    # Parse blocks
    aopcr_blocks = []
    best_block = None
    coverage_blocks = []

    # (파싱 로직)

    return {
        "best": best_block,          # {"Bag_Acc": 0.85, "Bag_F1": 0.82, ...}
        "aopcr": aopcr_blocks[-1],   # {"weighted": 0.45, "mean": 0.42, "classes": [...]}
        "coverage": coverage_blocks   # [{"threshold": 0.9, "explanation": 0.32, ...}, ...]
    }

def collect_latest_per_dataset(base_dir: Path, model_name: str) -> Dict[str, Dict]:
    """
    각 dataset별로 가장 최근 exp_* 디렉토리 찾아서 결과 수집

    Returns:
        {
            "ArticularyWordRecognition": {"best": {...}, "aopcr": {...}, ...},
            "BasicMotions": {...},
            ...
        }
    """
```

**사용 예:**
```bash
# 모든 데이터셋의 최신 AOPCR 결과 수집
python collect_aopcr.py --model InceptBackbone

# 출력:
# Dataset: ArticularyWordRecognition
#   Bag_Acc=0.9833, Bag_F1=0.9833
#   AOPCR (weighted)=0.4523
#   Coverage@90%: Expl=0.3245, Rand=0.5678, Gain=-0.2433
#
# Dataset: BasicMotions
#   ...
```

---

### 6.2 move_latest_exps.py (현재 Repo만)

**목적:** 최신 실험 결과를 백업 디렉토리로 이동

```python
def move_latest_experiments(
    source_dir: Path,
    dest_dir: Path,
    model_pattern: str = "*",
    dataset_pattern: str = "*",
    dry_run: bool = False
):
    """
    savemodel/<model>/<dataset>/exp_* 중 가장 최근 것을 백업

    Example:
        savemodel/InceptBackbone/Cricket/exp_15/
        → backup/20260113/InceptBackbone/Cricket/exp_15/
    """

    for model_dir in source_dir.glob(model_pattern):
        for dataset_dir in model_dir.glob(dataset_pattern):
            exp_dirs = sorted(dataset_dir.glob("exp_*"))
            if not exp_dirs:
                continue

            # Get latest by modification time
            latest_exp = max(exp_dirs, key=lambda p: p.stat().st_mtime)

            # Destination
            backup_path = dest_dir / model_dir.name / dataset_dir.name / latest_exp.name

            if dry_run:
                print(f"Would move: {latest_exp} → {backup_path}")
            else:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(latest_exp), str(backup_path))
                print(f"Moved: {latest_exp} → {backup_path}")
```

---

### 6.3 mydataload.py 차이

**현재 Repo:**
```python
# Lines 155-157
Xtr, ytr, meta = load_classification(name='InsectWingbeat', split='train',
                                     extract_path='../timeclass/dataset/')
```

**비교 Repo:**
```python
# Lines 155-158
Xtr, ytr, meta = load_classification(
    name='InsectWingbeat', split='train',
    extract_path='../timeclass/dataset/',
    return_metadata=True  # ✅ 명시적으로 metadata 요청
)
```

**차이점:** `return_metadata=True` 인자 추가 (aeon 라이브러리 버전 차이로 인한 호환성 처리)

---

### 6.4 instance_val.py / instance_val_cl.py 차이

**동일한 패턴:**
- 비교 Repo: `return_metadata=True` 명시
- 현재 Repo: 인자 생략 (기본값 사용)

**이유:** aeon 라이브러리 업데이트로 인한 API 변경

---

## 7. Shell Script 차이

### 7.1 현재 Repo Scripts (9개)

**Baseline 모델 실행:**
```bash
# run_ed1nn.sh
python main_cl_fix_ambiguous.py \
    --model ED1NN \
    --dataset ArticularyWordRecognition \
    --datatype original \
    --ed1nn_normalize

# run_dtw1nn.sh
python main_cl_fix_ambiguous.py \
    --model DTW1NN_I \
    --dataset Cricket \
    --datatype original \
    --dtw_window 10 \
    --dtw_max_train 100 \
    --dtw_max_test 50

# run_milet.sh
python main_cl_fix_ambiguous.py \
    --model MILLET \
    --dataset BasicMotions \
    --datatype mixed \
    --millet_pooling conjunctive \
    --epoch 100

# run_mlstm_fcn.sh / run_os_cnn.sh
# (비슷한 패턴)
```

**AmbiguousMIL 변형:**
```bash
# run_ambi_cl.sh
python main_cl_fix_ambiguous.py \
    --model AmbiguousMIL \
    --dataset Epilepsy \
    --datatype mixed \
    --proto_loss_w 0.1 \
    --smooth_loss_w 0.01

# run_ambi_ramp.sh
python main_cl_fix_ramp.py \  # ← ramp 버전 사용
    --model AmbiguousMIL \
    --ramp_mode linear \
    --ramp_start_epoch 10
```

---

### 7.2 비교 Repo Scripts (24개)

**DDP (Distributed Data Parallel) 위주:**
```bash
# run_all_ddp20.sh
torchrun --nproc_per_node=2 main_cl_fix.py \
    --model HybridMIL \
    --dataset ArticularyWordRecognition \
    --datatype mixed \
    --ddp_percent 0.2

# run_all_ddp30.sh, run_all_ddp40.sh, run_all_ddp50.sh
# (유사, ddp_percent만 다름)

# run_all_ddp20_exp.sh
# (실험용: main_exp.py 사용)

# run_all_ddp20_record.sh
# (recording: main_cl_record.py 사용)

# run_all_ddp20_n20.sh
# (n=20 설정)
```

**ContextualMIL 실행:**
```bash
# run_contextual.sh
python main_cl_fix.py \
    --model ContextualMIL \
    --attn_layers 2 \
    --attn_heads 8 \
    --attn_dropout 0.1 \
    --use_pos_enc 1

# run_contextual_single.sh
# (단일 데이터셋 버전)
```

**Hybrid 모델:**
```bash
# run_hybrid.sh
python main_cl_fix.py \
    --model HybridMIL \
    --dataset NATOPS \
    --datatype mixed
```

**차이점:**
- 현재: Baseline 비교 위주
- 비교: DDP 실험, architectural variants 위주

---

## 8. 문서 파일 차이

### 8.1 현재 Repo 전용

**AOPCR_EXTENDED_GUIDE.md:**
- AOPCR 0-100% 지원 설명
- Coverage metrics 사용법
- Command-line 예제

**MILLET_PORT_CHANGES.md:**
- MILLET 포팅 과정 문서화
- API 변경사항
- 사용 예제

**environment.yml:**
- Conda 환경 전체 명시
- Python 3.8.18
- PyTorch, timm, aeon, dtaidistance, wandb 등

**cmd.txt:**
- 자주 사용하는 명령어 모음

**.gitignore:**
- `__pycache__/`, `wandb/`, `savemodel/`, `*.pyc` 등

---

### 8.2 비교 Repo 전용

**HYBRID_USAGE.md:**
- HybridMIL/ContextualMIL 사용법
- Dual pooling 전략 설명
- Self-attention 설정 가이드

---

## 9. 요약 비교표

| Feature | 현재 Repo | 비교 Repo |
|---------|-----------|-----------|
| **Baseline 모델** | ✅ MILLET, MLSTM_FCN, OS_CNN, ED_1NN, DTW_1NN | ❌ |
| **Experimental 모델** | ❌ | ✅ HybridMIL, ContextualMIL |
| **Reproducibility** | ✅ 강화 (TF32 disable, worker seeding) | ⚠️ 기본 |
| **AOPCR Range** | ✅ 0-100% (configurable) | ❌ 0-50% (fixed) |
| **Coverage Metrics** | ✅ Coverage@X% | ❌ |
| **Self-Attention** | ❌ | ✅ ContextualMIL |
| **Dual Pooling** | ❌ | ✅ HybridMIL |
| **DDP Scripts** | ⚠️ 소수 | ✅ 다수 (24개) |
| **Documentation** | ✅ AOPCR Extended Guide | ✅ Hybrid Usage |
| **Environment File** | ✅ environment.yml | ❌ |
| **Utility Scripts** | ✅ collect_aopcr, move_exps | ❌ |
| **Git Status** | ✅ 깔끔 (6 modified) | ⚠️ 지저분 (12 modified, 많은 untracked) |
| **Commit History** | ✅ 15+ commits, 최신 | ⚠️ 2 commits, 구버전 |

---

## 10. 권장 사항

### 10.1 현재 Repo 사용 시나리오

1. **Baseline 비교 연구**
   - ED_1NN, DTW_1NN으로 전통적 방법 대비
   - MILLET, MLSTM_FCN, OS_CNN으로 딥러닝 baseline 비교

2. **AOPCR 메트릭 연구**
   - Coverage metrics로 해석 효율성 분석
   - 0-100% full range 평가

3. **재현성 중시 실험**
   - seed_everything으로 완전한 재현성
   - Worker seeding으로 DataLoader 재현성

---

### 10.2 비교 Repo 사용 시나리오

1. **Architectural Variants 연구**
   - HybridMIL의 dual pooling 효과 검증
   - ContextualMIL의 self-attention 효과 분석

2. **분산 학습 실험**
   - 많은 DDP script로 대규모 실험

3. **Attention Mechanism 연구**
   - Positional encoding 효과
   - Multi-head attention layer 수 최적화

---

### 10.3 통합 권장 사항

**현재 Repo에 추가하면 좋을 것:**
1. HybridMIL/ContextualMIL 모델 포팅
2. 더 많은 DDP 실험 스크립트

**비교 Repo에서 가져올 것:**
1. Baseline 모델들 (MILLET 등)
2. AOPCR extended metrics
3. Reproducibility 강화 코드
4. environment.yml

**최종 결론:** 현재 Repo가 더 발전된 버전이며, 비교 Repo의 architectural variants만 선택적으로 통합하는 것이 최선.
