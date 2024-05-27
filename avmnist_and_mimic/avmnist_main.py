import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import LeNet, MLP, Constant
import torch
import argparse
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from common_fusions import Concat, LowRankTensorFusion
from training_structures.inter_modality import train as train_inter_modality
from training_structures.inter_modality import test as test_inter_modality
from training_structures.unimodal import train as train_unimodal
from training_structures.unimodal import test as test_unimodal
from training_structures.inter_and_intra_modality import train as train_inter_and_intra_modality
from training_structures.inter_and_intra_modality import test as test_inter_and_intra_modality
from training_structures.intra_modality import train as train_intra_modality
from training_structures.intra_modality import test as test_intra_modality
from training_structures.inter_modality import MMDL

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='inter_modality', help='model type', type=str)
parser.add_argument('--fusion_type', default='lrtf', help='fusion type', type=str)
parser.add_argument('--modality_num', default=0, help='modality number (0-2?)', type=int)
parser.add_argument('--seed', default=0, help='seed', type=int)
parser.add_argument('--robust_data', help='Use robust data for evaluation?', action='store_true')
parser.add_argument('--test', help='Test only', action='store_true')
args = parser.parse_args()

traindata, validdata, testdata = get_dataloader(
    './data/avmnist')
ckpt_dir = f'./ckpts/avmnist/{args.seed}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if args.model_type == "inter_modality":
    if not args.test:
        channels = 6
        if args.fusion_type == "lf":
            encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
            head = MLP(channels*40, 100, 10).cuda()

            fusion = Concat().cuda()
            train_inter_modality(encoders, fusion, head, traindata, validdata, 30,
                save_dir=f'{ckpt_dir}/{args.fusion_type}.pt', optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)
        elif args.fusion_type == "lrtf":
            encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
            head = MLP(channels*20, 100, 10).cuda()

            fusion = LowRankTensorFusion([channels*8, channels*32], channels*20, 40).cuda()

            train_inter_modality(encoders, fusion, head, traindata, validdata, 30,
                save_dir=f'{ckpt_dir}/{args.fusion_type}.pt', optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)


    print("Testing:")
    model = torch.load(f'{ckpt_dir}/{args.fusion_type}.pt').cuda()
    test_inter_modality(model, testdata, no_robust=True)

elif args.model_type == "unimodal":
    if not args.test:
        channels = 3
        encoders=[LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
        heads = [MLP(channels*8, 100, 10).cuda(), MLP(channels* 32, 100, 10).cuda()]
        train_unimodal(encoders, heads, traindata, validdata, 30, optimtype=torch.optim.SGD,
                       save_dir=ckpt_dir, lr=0.01, weight_decay=0.0001, modalnum=args.modality_num)

    print("Testing:")
    model = torch.load(f'{ckpt_dir}/unimodal_{args.modality_num}.pt')
    test_unimodal(model, testdata, modalnum=args.modality_num, no_robust=True)

elif args.model_type == "inter_and_intra_modality":
    channels = 3
    if not args.test:
        inter_model = torch.load(f'{ckpt_dir}/{args.fusion_type}.pt').cuda()
        unimodal_0 = torch.load(f'{ckpt_dir}/unimodal_0.pt')
        unimodal_1 = torch.load(f'{ckpt_dir}/unimodal_1.pt')
        
        train_inter_and_intra_modality(inter_model, [unimodal_0, unimodal_1], \
                    traindata, validdata, 30, save_dir=ckpt_dir, \
                    optimtype=torch.optim.SGD, lr=0.01, weight_decay=0.0001)

    print("Testing:")
    inter_model = torch.load(f'{ckpt_dir}/mm_cat.pt').cuda()
    unimodal_models = [torch.load(f'{ckpt_dir}/mm_unimodal_0.pt'), torch.load(f'{ckpt_dir}/mm_unimodal_1.pt')]
    test_inter_and_intra_modality(inter_model, unimodal_models, testdata, no_robust=True)

elif args.model_type == "intra_modality":
    channels = 3
    if not args.test:
        unimodal_0 = torch.load(f'{ckpt_dir}/unimodal_0.pt')
        unimodal_1 = torch.load(f'{ckpt_dir}/unimodal_1.pt')
        
        train_intra_modality([unimodal_0, unimodal_1], \
                    traindata, validdata, 30, save_dir=ckpt_dir, \
                    optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)

    print("Testing:")

    unimodal_models = [torch.load(f'{ckpt_dir}/ens_unimodal_0.pt'), torch.load(f'{ckpt_dir}/ens_unimodal_1.pt')] 
    test_intra_modality(unimodal_models, testdata, no_robust=True)
