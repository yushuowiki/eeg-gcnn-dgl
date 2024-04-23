# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from joblib import load
from util.EEGGraphDataset import EEGGraphDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

from sklearn import preprocessing

from model.MCANet import MCANet
from model.shallow_EEGGraphConvNet import EEGGraphConvNet
from model.deep_EEGGraphConvNet import DEEGGraphConvNet
from model.SAGENet import SAGENet

from model.GATNet import GATNet
from model.MGATNet import MGATNet
from model.GINNet import GINNet
from model.SGCNet import SGCNet

# from model.GCN2Net import GCN2Net   #无法运行
# from model.TWIRLSNet import TWIRLSNet #无法运行
# from model.TAGNet import TAGNet
# from model.APPNPNet import  APPNPNet #这个未完成
# from model.PNANet import PNANet 

from tqdm import tqdm



if __name__ == "__main__":
    # argparse commandline args
    parser = argparse.ArgumentParser(description='Execute training pipeline on a given train/val subjects')
    parser.add_argument('--num_feats', type=int, default=6, help='Number of features per node for the graph')
    parser.add_argument('--num_nodes', type=int, default=8, help='Number of nodes in the graph')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='index of GPU device that should be used for this run, defaults to 0.')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs used to train')
    parser.add_argument('--exp_name', type=str, default='default', help='Name for the test.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size. Default is 512.')
    
    parser.add_argument('--model', type=str, default='sage',
                        help='shallow,deep,mca,sage,gin,gat,mhgat')
    args = parser.parse_args()

    # set the random seed so that we can reproduce the results
    np.random.seed(42)
    torch.manual_seed(42)

    # use GPU when available
    _GPU_IDX = args.gpu_idx
    _DEVICE = torch.device(f'cuda:{_GPU_IDX}' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(_DEVICE)
        print(f' Using device: {_DEVICE} {torch.cuda.get_device_name(_DEVICE)}')

    # load patient level indices
    _DATASET_INDEX = pd.read_csv("master_metadata_index.csv")
#     print(_DATASET_INDEX)
    all_subjects = _DATASET_INDEX["patient_ID"].astype("str").unique()
    
    print(f"Subject list fetched! Total subjects are {len(all_subjects)}.")

#     test_indices = _DATASET_INDEX.index[_DATASET_INDEX["patient_ID"].astype("str").isin(["sub-032301"])].tolist()


    # retrieve inputs
    num_nodes = args.num_nodes
    _NUM_EPOCHS = args.num_epochs
    _EXPERIMENT_NAME = args.exp_name
    _BATCH_SIZE = args.batch_size
    num_feats = args.num_feats

    # set up input and targets from files
    memmap_x = f'psd_features_data_X'
    memmap_y = f'labels_y'
    x = load(memmap_x, mmap_mode='r')
    y = load(memmap_y, mmap_mode='r')

#  -----------------------------------------------
    # normalize psd features data
#     normd_x = []
#     from tqdm import tqdm
#     for i in tqdm(range(len(y))):
#         arr = x[i, :]
#         arr = arr.reshape(1, -1)
#         arr2 = preprocessing.normalize(arr)
#         arr2 = arr2.reshape(48)
#         normd_x.append(arr2)
#     
#     norm = np.array(normd_x)
#     np.savetxt("norm_x.csv", norm, delimiter=",")
# -------------------------------------------    
    print("Loading data...")
    norm = pd.read_csv("norm_x.csv", delimiter=",",header = None)
    norm = norm.to_numpy()
# -------------------------------------------        
    
    x = norm.reshape(len(y), 48)
    # map 0/1 to diseased/healthy
    label_mapping, y = np.unique(y, return_inverse=True)
    print(f"Unique labels 0/1 mapping: {label_mapping}")

    # split the dataset to train and test. The ratio of test is 0.3.
    train_and_val_subjects, heldout_subjects = train_test_split(all_subjects, test_size=0.3, random_state=42)

    
    # split the dataset using patient indices
    train_window_indices = _DATASET_INDEX.index[
        _DATASET_INDEX["patient_ID"].astype("str").isin(train_and_val_subjects)].tolist()
    heldout_test_window_indices = _DATASET_INDEX.index[
        _DATASET_INDEX["patient_ID"].astype("str").isin(heldout_subjects)].tolist()

    # define model, optimizer, scheduler  
        # choose model
    if args.model == 'shallow':
        model = EEGGraphConvNet(num_feats)
    elif args.model == 'deep':    
        model = DEEGGraphConvNet(num_feats)
    elif args.model == 'mca':
        model = MCANet(num_feats)
    elif args.model == 'sage':
        model = SAGENet(num_feats)
    elif args.model == 'gin':
        model = GINNet(num_feats)    
#     elif args.model == 'mgat':
#         model = MGATNet(num_feats)   
    elif args.model == 'gat':   
        model = GATNet(num_feats)  
    elif args.model == 'sgc':   
        model = SGCNet(num_feats)  
#     elif args.model == 'gcn2':   
#         model = GCN2Net(num_feats)
#     elif args.model == 'twi':   
#         model = TWIRLSNet(num_feats) 
#     elif args.model == 'tag':   
#         model = TAGNet(num_feats) 
#     elif args.model == 'pna':
#         model = PNANet(num_feats)   
           

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * 10 for i in range(1, 26)], gamma=0.1)

    model = model.to(_DEVICE).double()
    print("GPU:",next(model.parameters()).is_cuda)
    print(model)
    
#     for name in model.state_dict():
#         print(name)
    
    
    num_trainable_params = np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()])

    # Dataloader========================================================================================================

    # use WeightedRandomSampler to balance the training dataset
    NUM_WORKERS = 4

    labels_unique, counts = np.unique(y, return_counts=True)

    class_weights = np.array([1.0 / x for x in counts])
    # provide weights for samples in the training set only
    sample_weights = class_weights[y[train_window_indices]]   #根据个数计算不同类别样本的采样率
    # sampler needs to come up with training set size number of samples
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_window_indices), replacement=True
    )

    # train data loader
    train_dataset = EEGGraphDataset(
        x=x, y=y, num_nodes=num_nodes, indices=train_window_indices
    )

    train_loader = GraphDataLoader(
        dataset=train_dataset, batch_size=_BATCH_SIZE,
        sampler=weighted_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # this loader is used without weighted sampling, to evaluate metrics on full training set after each epoch
    train_metrics_loader = GraphDataLoader(
        dataset=train_dataset, batch_size=_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # test data loader
    test_dataset = EEGGraphDataset(
        x=x, y=y, num_nodes=num_nodes, indices=heldout_test_window_indices
    )

    test_loader = GraphDataLoader(
        dataset=test_dataset, batch_size=_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    auroc_train_history = []
    auroc_test_history = []
    balACC_train_history = []
    balACC_test_history = []
    ACC_train_history = []
    ACC_test_history = []
    loss_train_history = []
    loss_test_history = []
    
    pre_test_his = []
    recall_test_his = []
    f1_test_his = []

    # training=========================================================================================================
    for epoch in tqdm(range(_NUM_EPOCHS)):
        model.train()
        train_loss = []

        for batch_idx, batch in enumerate(train_loader):
  
            # send batch to GPU
            g, dataset_idx, y = batch
            g_batch = g.to(device=_DEVICE, non_blocking=True)
            y_batch = y.to(device=_DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # forward pass
            outputs = model(g_batch)
            
            loss = loss_function(outputs, y_batch)
            train_loss.append(loss.item())

            # backward pass
            loss.backward()
            optimizer.step()      
            # 2023-2-24  =================================
#             heatmaps = []
#             if epoch % 10 == 0:
#                 index =  ((dataset_idx == 1000).nonzero(as_tuple=True)[0])
#                 if index.shape[0] != 0:
#                     index = index[0]
#                     if epoch == 0:
#                         dis = g_batch.edata['dis'].flatten()[64*index:64*(index+1)]
#                         dis = dis.detach().cpu().numpy()
#                         spec = g_batch.edata['spec'].flatten()[64*index:64*(index+1)]
#                         spec = spec.detach().cpu().numpy()
#                         heatmaps.append(dis)
#                         heatmaps.append(spec)
#                     edge_weight = model.get_dweight(g_batch).flatten()[64*index:64*(index+1)]
#                     edge_weight = edge_weight.detach().cpu().numpy()
#                     print(edge_weight)
#                     heatmaps.append(edge_weight)
#             np.savetxt("heatmaps_array.txt", np.array(heatmaps))   
            # End  =================================
          
        # update learning rate
        scheduler.step()

    # evaluate model after each epoch for train-metric data============================================================
        model.eval()
        with torch.no_grad():
            y_probs_train = torch.empty(0, 2).to(_DEVICE)
            y_true_train, y_pred_train = [], []

            for i, batch in enumerate(train_metrics_loader):
                g, dataset_idx, y = batch
                g_batch = g.to(device=_DEVICE, non_blocking=True)
                y_batch = y.to(device=_DEVICE, non_blocking=True)

                # forward pass
                outputs = model(g_batch)

                _, predicted = torch.max(outputs.data, 1)
                y_pred_train += predicted.cpu().numpy().tolist()
                # concatenate along 0th dimension
                y_probs_train = torch.cat((y_probs_train, outputs.data), 0)
                y_true_train += y_batch.cpu().numpy().tolist()

        # returning prob distribution over target classes, take softmax over the 1st dimension
        y_probs_train = nn.functional.softmax(y_probs_train, dim=1).cpu().numpy()
        y_true_train = np.array(y_true_train)

    # evaluate model after each epoch for validation data ==============================================================
        y_probs_test = torch.empty(0, 2).to(_DEVICE)
        y_true_test, minibatch_loss, y_pred_test = [], [], []

        embeddings = []
        for i, batch in enumerate(test_loader):
            g, dataset_idx, y = batch
            g_batch = g.to(device=_DEVICE, non_blocking=True)
            y_batch = y.to(device=_DEVICE, non_blocking=True)

            # forward pass
            outputs = model(g_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_test += predicted.cpu().numpy().tolist()

            loss = loss_function(outputs, y_batch)
            minibatch_loss.append(loss.item())
            y_probs_test = torch.cat((y_probs_test, outputs.data), 0)
            y_true_test += y_batch.cpu().numpy().tolist()

        # returning prob distribution over target classes, take softmax over the 1st dimension
        y_probs_test = torch.nn.functional.softmax(y_probs_test, dim=1).cpu().numpy()
        y_true_test = np.array(y_true_test)
# ---------------------------------------------------------------------
        # record training auroc and testing auroc
        auroc_train_history.append(roc_auc_score(y_true_train, y_probs_train[:, 1]))
        auroc_test_history.append(roc_auc_score(y_true_test, y_probs_test[:, 1]))

        # record training balanced accuracy and testing balanced accuracy
        balACC_train_history.append(balanced_accuracy_score(y_true_train, y_pred_train))
        balACC_test_history.append(balanced_accuracy_score(y_true_test, y_pred_test))

        ACC_train_history.append(accuracy_score(y_true_train, y_pred_train))
        ACC_test_history.append(accuracy_score(y_true_test, y_pred_test))


        # LOSS - epoch loss is defined as mean of minibatch losses within epoch
        loss_train_history.append(np.mean(train_loss))
        loss_test_history.append(np.mean(minibatch_loss))
        
        p, r, f1,_ =  precision_recall_fscore_support(y_true_test, y_pred_test,average='micro')
        pre_test_his.append(p)
        recall_test_his.append(r)
        f1_test_his.append(f1)

# -----------------------------------------------------------------------

        # print the metrics
        print("Train loss: {}, test loss: {}".format(loss_train_history[-1], loss_test_history[-1]))
        print("Train AUC: {}, test AUC: {}".format(auroc_train_history[-1], auroc_test_history[-1]))
        print("Train Bal.ACC: {}, test Bal.ACC: {}".format(balACC_train_history[-1], balACC_test_history[-1]))
#         save results ,直接覆盖重新写入。
        data = {'auroc_train':auroc_train_history,
                'auroc_test':auroc_test_history,
                'balACC_train':balACC_train_history,
                'balACC_test':balACC_test_history,
                'ACC_train':ACC_train_history,
                'ACC_test':ACC_test_history,
                'loss_train':loss_train_history,
                'loss_test':loss_test_history,
                'pre_test_his':pre_test_his,
                'recall_test_his':recall_test_his,
                'f1_test_his':f1_test_his}
        df = pd.DataFrame(data)
        df.to_csv("./log/"+args.model+"_results.csv",index_label = "Epoch")



        # save model from each epoch====================================================================================
        state = {
            'epochs': _NUM_EPOCHS,
            'experiment_name': _EXPERIMENT_NAME,
            'model_description': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f"{_EXPERIMENT_NAME}_Epoch_{epoch}.ckpt")
        
        
#         -------------------------------------------------------
#     data = {'auroc_train':auroc_train_history,
#             'auroc_test':auroc_test_history,
#             'balACC_train':balACC_train_history,
#             'balACC_test':balACC_test_history,
#             'loss_train':loss_train_history,
#             'loss_test':loss_test_history}
#     df = pd.DataFrame(data)
#     df.to_csv("./log/"+args.model+"_results.csv",index_label = "Epoch")
        
        
        
