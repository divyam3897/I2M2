import sys
import os
import torch
import argparse
from torch import nn

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.common_models import MLP, GRU, GRUWithLinear # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from common_fusions import Concat # noqa
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
parser.add_argument('--task', default='mimic_7', help='task type', type=str)
parser.add_argument('--modality_num', default=0, help='modality number (0-2?)', type=int)
parser.add_argument('--seed', default=1, help='seed', type=int)
parser.add_argument('--robust_data', help='Use robust data for evaluation?', action='store_true')
parser.add_argument('--test', help='Test only', action='store_true')
args = parser.parse_args()

# get dataloader for icd9 classification task 7
if args.task == "mortality":
    traindata, validdata, testdata = get_dataloader(
        -1, imputed_path='./data/mimic/im.pk')
elif args.task == "mimic_1":
    traindata, validdata, testdata = get_dataloader(
        1, imputed_path='./data/mimic/im.pk')
elif args.task == "mimic_7":
    traindata, validdata, testdata = get_dataloader(
        7, imputed_path='./data/mimic/im.pk')

ckpt_dir = f'./ckpts/mimic/{args.task}/{args.seed}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if args.model_type == "unimodal":
    if args.task == "mortality":
        if not args.test:
            encoders = [MLP(5, 10, 10).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            heads = [MLP(10, 40, 6, dropout=False).cuda(), MLP(360, 40, 6, dropout=False).cuda()] 
            train_unimodal(encoders, heads, traindata, validdata, 20, \
                           save_dir=ckpt_dir, auprc=False, modalnum=args.modality_num)

        model = torch.load(f'{ckpt_dir}/unimodal_{args.modality_num}.pt')
        test_unimodal(model, testdata, dataset='mortality', auprc=False, modalnum=args.modality_num)
    elif args.task == "mimic_1":
        if not args.test:
            encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 
            train_unimodal(encoders, heads, traindata, validdata, 20, \
                           save_dir=ckpt_dir,auprc=True, modalnum=args.modality_num)

        model = torch.load(f'{ckpt_dir}/unimodal_{args.modality_num}.pt')
        test_unimodal(model, testdata, dataset='mimic_1', auprc=True, modalnum=args.modality_num)
    elif args.task == "mimic_7":
        if not args.test:
            encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 
            train_unimodal(encoders, heads, traindata, validdata, 20, \
                           save_dir=ckpt_dir, auprc=True, modalnum=args.modality_num)
        
        model = torch.load(f'{ckpt_dir}/unimodal_{args.modality_num}.pt')
        test_unimodal(model, testdata, dataset='mimic_7', auprc=True, modalnum=args.modality_num)


if args.model_type == "inter_modality":
    if args.task == "mortality":
        if not args.test:
            encoders = [MLP(5, 10, 10).cuda(), GRU(
                12, 30, flatten=True, batch_first=True).cuda()]
            head = MLP(730, 40, 6, dropout=False).cuda()
            fusion = Concat().cuda()
            train_inter_modality(encoders, fusion, head, traindata, validdata, 20, save_dir=ckpt_dir + 'lf.pt', auprc=False)

        print("Testing: ")
        model = torch.load(f'{ckpt_dir}/lf.pt').cuda()
        test_inter_modality(model, testdata, dataset='mimic mortality', auprc=False)
    elif args.task == "mimic_1":
        if not args.test:
            encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(
                    12, 30, dropout=False, batch_first=True).cuda()]
            head = MLP(730, 40, 2, dropout=False).cuda()
            fusion = Concat().cuda()
            train_inter_modality(encoders, fusion, head, traindata, validdata, 20, save_dir=ckpt_dir + 'lf.pt', auprc=True)

        print("Testing: ")
        model = torch.load(f'{ckpt_dir}/lf.pt').cuda()
        test_inter_modality(model, testdata, dataset='mimic mortality', auprc=True)
    elif args.task == "mimic_7":
        if not args.test:
            encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(
                    12, 30, dropout=False, batch_first=True).cuda()]
            head = MLP(730, 40, 2, dropout=False).cuda()
            fusion = Concat().cuda()
            train_inter_modality(encoders, fusion, head, traindata, validdata, 20, save_dir=ckpt_dir + '/lf.pt', auprc=True)

        print("Testing: ")
        model = torch.load(f'{ckpt_dir}/lf.pt').cuda()
        test_inter_modality(model, testdata, dataset='mimic 7', auprc=True)

if args.model_type == "inter_and_intra_modality":
    if args.task == "mortality":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 6, dropout=False).cuda(), MLP(360, 40, 6, dropout=False).cuda()] 
        
            md_encoders = [MLP(5, 10, 10).cuda(), GRU(
                12, 30, flatten=True, batch_first=True).cuda()]
            md_head = MLP(730, 40, 6, dropout=False).cuda()
            fusion = Concat().cuda()
        
            md_model = MMDL(md_encoders, fusion, md_head, has_padding=False).cuda()
            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))

            train_inter_and_intra_modality(md_model, unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=False)
        
        print("Testing:")
        md_model = torch.load(f'{ckpt_dir}/mm_cat.pt').cuda()

        unimodal_models = [torch.load(f'{ckpt_dir}/mm_unimodal_0.pt'), torch.load(f'{ckpt_dir}/mm_unimodal_1.pt')]
        test_inter_and_intra_modality(md_model, unimodal_models, testdata, dataset='mortality', auprc=False)
    elif args.task == "mimic_1":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 
            md_encoders = [MLP(5, 10, 10).cuda(), GRU(
                12, 30, flatten=True, batch_first=True).cuda()]
            md_head = MLP(730, 40, 2, dropout=False).cuda()
            fusion = Concat().cuda()
        
            md_model = MMDL(md_encoders, fusion, md_head, has_padding=False).cuda()
            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))

            train_inter_and_intra_modality(md_model, unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=True)

        print("Testing:")
        md_model = torch.load(f'{ckpt_dir}/mm_cat.pt').cuda()

        unimodal_models = [torch.load(f'{ckpt_dir}/mm_unimodal_0.pt'), torch.load(f'{ckpt_dir}/mm_unimodal_1.pt')] 
        test_inter_and_intra_modality(md_model, unimodal_models, testdata, dataset='mimic_1', auprc=True)
    elif args.task == "mimic_7":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 
            md_encoders = [MLP(5, 10, 10).cuda(), GRU(
                12, 30, flatten=True, batch_first=True).cuda()]
            md_head = MLP(730, 40, 2, dropout=False).cuda()
            fusion = Concat().cuda()
        
            md_model = MMDL(md_encoders, fusion, md_head, has_padding=False).cuda()
            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))

            train_inter_and_intra_modality(md_model, unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=True)

        print("Testing:")
        md_model = torch.load(f'{ckpt_dir}/mm_cat.pt').cuda()
        unimodal_models = [torch.load(f'{ckpt_dir}/mm_unimodal_0.pt'), torch.load(f'{ckpt_dir}/mm_unimodal_1.pt')] 
        
        test_inter_and_intra_modality(md_model, unimodal_models, testdata, dataset='mimic_7', auprc=True)

elif args.model_type == "intra_modality":
    if args.task == "mortality":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 6, dropout=False).cuda(), MLP(360, 40, 6, dropout=False).cuda()] 

            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))
        
            train_intra_modality(unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=False)
        
        print("Testing:")
        unimodal_models = [torch.load(f'{ckpt_dir}/ens_unimodal_0.pt'), torch.load(f'{ckpt_dir}/ens_unimodal_1.pt')] 
        test_intra_modality(unimodal_models, testdata, dataset='mortality', auprc=False)
    elif args.task == "mimic_1":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 
            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))

            train_intra_modality(unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=True)

        print("Testing:")
        unimodal_models = [torch.load(f'{ckpt_dir}/ens_unimodal_0.pt'), torch.load(f'{ckpt_dir}/ens_unimodal_1.pt')] 
        test_intra_modality(unimodal_models, testdata, dataset='mimic_1', auprc=True)
    elif args.task == "mimic_7":
        if not args.test:
            unimodal_encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).cuda()]
            unimodal_heads = [MLP(10, 40, 2, dropout=False).cuda(), MLP(360, 40, 2, dropout=False).cuda()] 

            unimodal_models = []
            for i in range(2):
                unimodal_models.append(nn.Sequential(unimodal_encoders[i], unimodal_heads[i]))
            train_intra_modality(unimodal_models, traindata, validdata, 20, save_dir=ckpt_dir, auprc=True)
        
        print("Testing:")
        unimodal_models = [torch.load(f'{ckpt_dir}/ens_unimodal_0.pt'), torch.load(f'{ckpt_dir}/ens_unimodal_1.pt')] 
        test_intra_modality(unimodal_models, testdata, dataset='mimic_7', auprc=True)
