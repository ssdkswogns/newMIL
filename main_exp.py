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

from models.timemil import TimeMIL, newTimeMIL

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

def train(trainloader, milnet, criterion, optimizer, epoch,args):
    milnet.train()
    total_loss = 0


    for batch_id, (feats, label) in enumerate(trainloader):
    #     bag_feats = feats.cuda()
    #     bag_label = label.cuda()

        bag_feats = feats.cuda()
        bag_label = label.cuda()
        
        # print(bag_feats.shape)
        
        # window-based random masking
        if args.dropout_patch>0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(bag_feats)//10)
            
            for idx in selecy_window_indx:
                bag_feats[:,idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
        
   

        optimizer.zero_grad()
   
        if epoch<args.epoch_des:
            bag_prediction  = milnet(bag_feats,warmup = True)
        else:
            bag_prediction  = milnet(bag_feats,warmup = False)
       
        bag_loss = criterion(bag_prediction, bag_label)
       
        if True:
            loss = bag_loss 
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), bag_loss.item(),loss.item()))
        
        
     
        loss.backward()
        
    
        # avoid the overfitting by using gradient clip
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()

        # total_loss = total_loss + loss.item()
        total_loss += bag_loss.item()
      

    return total_loss / len(trainloader)

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def test(testloader, milnet, criterion, args, threshold: float = 0.5):
    milnet.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()

            logits = milnet(bag_feats)                   # [B, C]
            loss   = criterion(logits, bag_label)        # BCEWithLogitsLoss
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]
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

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
    }
    return total_loss / len(testloader), results

def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="PenDigits", type=str, help='dataset ')
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
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
   
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')               
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir+'InceptBackbone'
    
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)
    


    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL
    # scaler = GradScaler()
    
    ###################################
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
       
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        
        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in
        
        
        
        
        print(f'max lenght {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}' )
        
    else:
        # 원본 로드
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train',extract_path='./data')
        Xte, yte, _   = load_classification(name=args.dataset, split='test',extract_path='./data')

        word_to_idx = {cls:i for i, cls in enumerate(meta['class_values'])}
        ytr_idx = torch.tensor([word_to_idx[i] for i in ytr], dtype=torch.long)
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

        Xtr = torch.from_numpy(Xtr).permute(0,2,1).float()  # [N,T,D]
        Xte = torch.from_numpy(Xte).permute(0,2,1).float()

        num_classes = len(meta['class_values'])
        args.num_classes = num_classes
        L_in = Xtr.shape[-1]
        seq_len = max(21, Xte.shape[1])   # 모델 max_seq_len 용

        # 혼합 비율 설정: 원본 40%, 2클 40%, 3클 20%
        mix_probs = {'orig': 0.4, 2: 0.4, 3: 0.2}
        SYN_TOTAL_LEN = seq_len  # 필요시 2500 등으로 고정 가능

        trainset = MixedSyntheticBags(
            X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
            total_bags=max(2000, len(Xtr)),
            probs=mix_probs, total_len=SYN_TOTAL_LEN,
            min_seg_len=32, ensure_distinct_classes=True, seed=args.seed
        )

        testset = MixedSyntheticBags(
            X=Xte, y_idx=yte_idx, num_classes=num_classes,
            total_bags=max(1000, len(Xte)),
            probs=mix_probs, total_len=SYN_TOTAL_LEN,
            min_seg_len=32, ensure_distinct_classes=True, seed=args.seed+1
        )

        args.feats_size = L_in
        print(f'num class: {args.num_classes}')
        print(f'total_len: {SYN_TOTAL_LEN}')
    
    
    
    
    # <------------- define MIL network ------------->
   
    if args.model =='TimeMIL':
        milnet = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    elif args.model == 'newTimeMIL':
        milnet = newTimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    else:
        raise Exception("Model not available")
    
    if  args.optimizer == 'adamw':
    # optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer, alpha=0.5, k=5)    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer =Lookahead(optimizer) 
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer, alpha=0.5, k=5) 
    
    # elif args.optimizer == 'adamp':
    #     optimizer = AdamP(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer =Lookahead(optimizer, alpha=0.5, k=5) 
    
    
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

    trainloader = DataLoader(trainset, batch, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # if args.batchsize==1:
    #     testloader = DataLoader(testset, args.batchsize, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # else:
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    for epoch in range(1, args.num_epochs + 1):

        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch,args) # iterate all bags
        
      
        test_loss_bag, results = test(testloader, milnet, criterion, args, threshold=0.5)
    
        logger.info(
            ('Epoch [%d/%d] train loss: %.4f test loss: %.4f | '
            'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
            'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
            (epoch, args.num_epochs, train_loss_bag, test_loss_bag,
            results["f1_micro"], results["f1_macro"],
            results["p_micro"],  results["p_macro"],
            results["r_micro"],  results["r_macro"],
            results["roc_auc_macro"], results["mAP_macro"])
        )

        # 기존 current_score = avg_score
        # 멀티라벨에선 보통 F1_micro나 mAP_macro를 베스트 기준으로 사용
        current_score = results["mAP_macro"]  # 또는 results["mAP_macro"]

        if current_score >= best_score:
            
            results_best = results
            
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, f'best_{args.model}.pth')

            torch.save(milnet.state_dict(), save_name)
            #torch.save(milnet, save_name)
            logger.info('Best model saved at: ' + save_name)
            # logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
    best = results_best  # test()에서 얻은 dict
    logger.info(
        ('Best Results | '
        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
        (best["f1_micro"], best["f1_macro"],
        best["p_micro"],  best["p_macro"],
        best["r_micro"],  best["r_macro"],
        best["roc_auc_macro"], best["mAP_macro"])
    )


        # if args.weight_div>0:
        #     if epoch%10==0:
        #         print('--------------------Clustering--------------------\n')
        #         cluster_idx_dict = pre_cluter(trainloader, milnet, criterion, optimizer, args,init= False)
        #         print('--------------------Clustering finished--------------------\n')

if __name__ == '__main__':
    main()
