# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

# from models.dropout import LinearScheduler
from aeon.datasets import load_classification
from syntheticdataset import *
from utils import *
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from mydataload import loadorean
import random

# from sample_method import rd_torch,dpp
import random
# from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings

from models.timemil import TimeMIL, newTimeMIL, AmbiguousMIL

# Suppress all warnings
warnings.filterwarnings("ignore")

seed = 42

random.seed(seed)             # python random
np.random.seed(seed)          # numpy random
torch.manual_seed(seed)       # CPU
torch.cuda.manual_seed(seed)  # GPU 단일
torch.cuda.manual_seed_all(seed)  # multi-GPU

# 재현성을 위한 옵션들
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def test(testloader, milnet, criterion, args, class_names, threshold: float = 0.5):
    milnet.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    per_class_correct = None
    per_class_total   = None
    total_correct = 0
    total_count   = 0

    with torch.no_grad():
        for batch_id, (feats, label, y_inst) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            
            if args.model == 'AmbiguousMIL':
                logits, instance_logits, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
            else:
                logits, x, attn_layer1, attn_layer2 = milnet(bag_feats)
            loss   = criterion(logits, bag_label)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]
            B, T, C = y_inst.shape

            attn_cls = attn_layer2[:,:,:C,C:]
            attn_mean = attn_cls.mean(dim=1)

            if args.model == 'AmbiguousMIL':
                pred_inst = torch.argmax(instance_logits, dim=2).cpu()
            else:
                pred_inst = torch.argmax(attn_mean, dim=1).cpu()

            y_inst_label = torch.argmax(y_inst, dim=2).cpu()

            correct = (pred_inst == y_inst_label).sum().item()
            count = pred_inst.numel()

            total_correct += correct
            total_count   += count

            if per_class_correct is None:
                per_class_correct = torch.zeros(C, dtype=torch.long)
                per_class_total = torch.zeros(C, dtype=torch.long)

            pb = pred_inst.view(-1)
            tb = y_inst_label.view(-1)
            for c_id in range(C):
                mask = (tb == c_id)
                if mask.any():
                    per_class_correct[c_id] += (pb[mask] == tb[mask]).sum().cpu()
                    per_class_total[c_id]   += mask.sum().cpu()

            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f'
                             % (batch_id, len(testloader), loss.item()))

    y_true = np.vstack(all_labels)   # [N, C], multi-hot
    y_prob = np.vstack(all_probs)    # [N, C], sigmoid prob
    y_pred = (y_prob >= threshold).astype(np.int32)  # [N, C]

    # 멀티라벨 지표
    # (정의상 일부 샘플이 all-zero 또는 all-one이 될 수 있으므로 zero_division=0 권장)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    p_micro  = precision_score(y_true, y_pred, average='micro', zero_division=0)
    p_macro  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
    r_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # ROC-AUC / mAP(=Average Precision) — 클래스별 계산 후 macro 평균
    roc_list, ap_list = [], []
    for c in range(y_true.shape[1]):
        # 양/음 라벨이 모두 있을 때만 계산
        if len(np.unique(y_true[:, c])) == 2:
            try:
                roc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
                ap_list.append(average_precision_score(y_true[:, c], y_prob[:, c]))
            except Exception:
                pass
    roc_macro = float(np.mean(roc_list)) if roc_list else 0.0
    ap_macro  = float(np.mean(ap_list))  if ap_list  else 0.0

        # 결과 산출
    timestep_acc = float(total_correct / total_count) if total_count > 0 else 0.0
    per_class_acc = []
    for c_id in range(per_class_total.numel()):
        tot = int(per_class_total[c_id].item())
        cor = int(per_class_correct[c_id].item())
        acc = float(cor / tot) if tot > 0 else 0.0
        label = class_names[c_id] if (class_names is not None and c_id < len(class_names)) else f"class-{c_id}"
        per_class_acc.append({"class_id": c_id, "label": label, "count": tot, "correct": cor, "acc": acc})

    inst_acc = {
        "timestep_accuracy_overall": timestep_acc,
        "per_class": per_class_acc
    }

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
    }
    return total_loss / len(testloader), results, inst_acc

def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="BasicMotions", type=str, help='dataset ')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-3, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--model', default='TimeMIL', type=str, help='MIL model')
    parser.add_argument('--model_path', default='./savemodel/InceptBackbone/BasicMotions/exp_7/weights/best_newTimeMIL.pth', type=str, help='Target model path for evaluation')
   
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')               
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL
    # scaler = GradScaler()
    
    ###################################
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
       
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        
        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in
        
        print(f'max lenghth{seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}' )
        
    else:
        # 원본 로드
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train',extract_path='./data')
        Xte, yte, _   = load_classification(name=args.dataset, split='test',extract_path='./data')

        word_to_idx = {cls:i for i, cls in enumerate(meta['class_values'])}
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

        Xtr = torch.from_numpy(Xtr).permute(0,2,1).float()  # [N,T,D]
        Xte = torch.from_numpy(Xte).permute(0,2,1).float()

        num_classes = len(meta['class_values'])
        class_values = meta.get('class_values', None)

        if class_values is not None:
            class_names = list(class_values)
            word_to_idx = {cls:i for i, cls in enumerate(class_values)}
            yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)
        
        args.num_classes = num_classes
        L_in = Xtr.shape[-1]
        seq_len = max(21, Xte.shape[1])   # 모델 max_seq_len 용

        # 혼합 비율 설정: 원본 40%, 2클 40%, 3클 20%
        mix_probs = {'orig': 0.4, 2: 0.4, 3: 0.2}
        SYN_TOTAL_LEN = seq_len  # 필요시 2500 등으로 고정 가능

        testset = MixedSyntheticBags(
            X=Xte, y_idx=yte_idx, num_classes=num_classes,
            total_bags=max(1000, len(Xte)),
            probs=mix_probs, total_len=SYN_TOTAL_LEN,
            min_seg_len=32, ensure_distinct_classes=True, seed=args.seed+1,
            return_instance_labels=True
        )

        args.feats_size = L_in
        print(f'num class: {args.num_classes}')
        print(f'total_len: {SYN_TOTAL_LEN}')
    
    
    # <------------- define MIL network ------------->
   
    if args.model =='TimeMIL':
        milnet = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len, is_instance= True).cuda()
    elif args.model == 'newTimeMIL':
        milnet = newTimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len, is_instance= True).cuda()
    elif args.model == 'AmbiguousMIL':
        milnet = AmbiguousMIL(args.feats_size,mDim=args.embed,n_classes =num_classes, dropout=args.dropout_node, max_seq_len = seq_len, is_instance=True).cuda()
    else:
        raise Exception("Model not available")
    
    
    if args.dataset in ['ArticularyWordRecognition','BasicMotions','Cricket','ERing','HandMovementDirection','LSST','PhonemeSpectra','RacketSports','SelfRegulationSCP1','SelfRegulationSCP2','UWaveGestureLibrary']:
        batch = 64
    elif args.dataset in ['AtrialFibrillation','FingerMovements','Handwriting','Heartbeat','MotorImagery']:
        batch = 32
    elif args.dataset in ['DuckDuckGeese','EthanolConcentration','NATOPS','JapaneseVowels']:
        batch = 16
    elif args.dataset in ['Epilepsy','FaceDetection','Libras','PEMS-SF', 'StandWalkJump']:
        batch = 8
    elif args.dataset in ['StandWalkJump']:
        batch = 1
    elif args.dataset in ['PenDigits']:
        batch = 128
    else:
        batch = args.batchsize

    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    milnet.load_state_dict(torch.load(args.model_path))
        
    test_loss_bag, results, inst_acc = test(testloader, milnet, criterion, args, class_names = class_names, threshold=0.5)
    print(results)
    print(inst_acc)

if __name__ == '__main__':
    main()
