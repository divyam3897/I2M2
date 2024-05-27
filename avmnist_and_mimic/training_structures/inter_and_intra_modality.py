
"""Implements supervised learning training procedures."""
import torch
from torch import nn
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.functional import calibration_error 
from .inter_modality import deal_with_objective
#import pdb

softmax = nn.Softmax()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train(
        md_model, unimodal_models, train_dataloader, valid_dataloader, total_epochs, save_dir, additional_optimizing_modules=[], is_packed=False,
        early_stop=True, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=False):
    """
    Handle running a simple supervised training loop.
    
    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    total_parameters = list(md_model.parameters()) + list(unimodal_models[0].parameters()) + list(unimodal_models[1].parameters())
    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in total_parameters if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0
        bestauprc = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            md_model.train()
            for unimodal_model in unimodal_models:
                unimodal_model.train()
            totalloss = 0.0
            totals = 0
            all_outputs, all_true = [], []
            all_outputs_1, all_outputs_2, all_outputs_cat = [], [], []
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        md_model.train()
                        for unimodal_model in unimodal_models:
                            unimodal_model.train()
                        out = model.inter * model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                        for idx, uni_model in enumerate(unimodal_models):
                            out += (model.intra * uni_model(j[idx].float().to(device)))
                        
                        out /= (len(unimodal_models) + 1)

                else:
                    md_model.train()
                    for unimodal_model in unimodal_models:
                        unimodal_model.train()
                    output_cat = md_model([_processinput(i).to(device) for i in j[:-1]])
                    output_1 = unimodal_models[0](j[0].float().to(device))
                    output_2 = unimodal_models[1](j[1].float().to(device))
                    output_num = torch.log_softmax(output_1, dim=-1) + \
                                 torch.log_softmax(output_2, dim=-1) + \
                                 torch.log_softmax(output_cat, dim=-1)

                    output_den = torch.logsumexp(output_num, dim=-1)
                    out = output_num - output_den.unsqueeze(1) 
                    
                    all_outputs_1.append(output_1)
                    all_outputs_2.append(output_2)
                    all_outputs_cat.append(output_cat)
                    all_outputs.append(out)

                    all_true.append(j[-1])

                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(total_parameters, clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))

            all_true = torch.cat(all_true, 0)
            all_outputs_1 = torch.cat(all_outputs_1, 0)
            all_outputs_2 = torch.cat(all_outputs_2, 0)
            all_outputs_cat = torch.cat(all_outputs_cat, 0)
            all_outputs = torch.cat(all_outputs, 0)

            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            md_model.eval()
            for unimodal_model in unimodal_models:
                unimodal_model.eval()
            with torch.no_grad():
                totalloss = 0.0
                total_batches = 0.
                pred = []
                true = []
                pts = []
                all_outputs, all_true = [], []
                all_outputs_1, all_outputs_2, all_outputs_cat = [], [], []
                for j in valid_dataloader:
                    if is_packed:
                        out = md_model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        output_cat = md_model([_processinput(i).to(device) for i in j[:-1]])
                        output_1 = unimodal_models[0](j[0].float().to(device))
                        output_2 = unimodal_models[1](j[1].float().to(device))
                        output_num = torch.log_softmax(output_1, dim=-1) + \
                                 torch.log_softmax(output_2, dim=-1) + \
                                 torch.log_softmax(output_cat, dim=-1)

                        output_den = torch.logsumexp(output_num, dim=-1)
                        out = output_num - output_den.unsqueeze(1) 

                        all_outputs_1.append(output_1)
                        all_outputs_2.append(output_2)
                        all_outputs_cat.append(output_cat)
                        all_outputs.append(out)

                        all_true.append(j[-1])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    totalloss += loss*len(j[-1])
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        sm = out
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                all_true = torch.cat(all_true, 0)
                all_outputs_1 = torch.cat(all_outputs_1, 0)
                all_outputs_2 = torch.cat(all_outputs_2, 0)
                all_outputs_cat = torch.cat(all_outputs_cat, 0)
                all_outputs = torch.cat(all_outputs, 0)

                if auprc:
                    curr_auprc = AUPRC(pts) 
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " acc: "+str(acc))
                if not auprc and (acc > bestacc):
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(md_model, f'{save_dir}/mm_cat.pt')
                    for i in range(len(unimodal_models)):
                        torch.save(unimodal_models[i], f'{save_dir}/mm_unimodal_{i}.pt')
                elif auprc and (curr_auprc > bestauprc):
                    patience = 0
                    bestauprc = curr_auprc
                    print("Saving Best")
                    torch.save(md_model, f'{save_dir}/mm_cat.pt')
                    for i in range(len(unimodal_models)):
                        torch.save(unimodal_models[i], f'{save_dir}/mm_unimodal_{i}.pt') 
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > 10:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
    if track_complexity:
        all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        _trainprocess()


def single_test(
        md_model, unimodal_models, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred, true, all_outputs = [], [], []
        all_outputs_cat, all_outputs_1, all_outputs_2 = [], [], []
        pts = []

        md_model.eval()
        for model in unimodal_models:
            model.eval()
        for j in test_dataloader:
            if is_packed:
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in j[0]], j[1]])
            else:
                output_cat = md_model([_processinput(i).to(device) for i in j[:-1]])
                output_1 = unimodal_models[0](j[0].float().to(device))
                output_2 = unimodal_models[1](j[1].float().to(device))
                output_num = (torch.log_softmax(output_1, dim=-1)) + \
                                (torch.log_softmax(output_2, dim=-1)) + \
                                (torch.log_softmax(output_cat, dim=-1))

                output_den = torch.logsumexp(output_num, dim=-1)
                out = output_num - output_den.unsqueeze(1) 

            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size())-1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            else:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            totalloss += loss*len(j[-1])
            if task == "classification":
                all_outputs.append(out)
                all_outputs_cat.append(output_cat)
                all_outputs_1.append(output_1)
                all_outputs_2.append(output_2)
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                sm = out
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        all_outputs = torch.cat(all_outputs, 0)
        all_outputs_cat = torch.cat(all_outputs_cat, 0)
        all_outputs_1 = torch.cat(all_outputs_1, 0)
        all_outputs_2 = torch.cat(all_outputs_2, 0)
        max_accuracy = accuracy(true, torch.argmax(all_outputs, 1))
        if auprc:
            sm = all_outputs
            pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            max_auprc = AUPRC(pts)
            print("AUPRC", max_auprc)

        totals = len(true)

        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}



def test(
        md_model, unimodal_models, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False):
    """
    Handle getting test results for a simple supervised training loop.
    
    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there   
    """
    if no_robust:
        def _testprocess():
            single_test(md_model, unimodal_models, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(_testprocess, [md_model, unimodal_models])
        return

    def _testprocess():
        single_test(md_model, unimodal_models, test_dataloaders_all[list(test_dataloaders_all.keys())[
                    0]][0], is_packed, criterion, task, auprc, input_to_float)
    all_in_one_test(_testprocess, [md_model, unimodal_models])
    