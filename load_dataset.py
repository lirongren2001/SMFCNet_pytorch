import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data  # DataLoader
from torch_geometric.loader import DataLoader
from glob import glob
import csv
import pickle
import numpy as np
import re
import pandas as pd
import scipy as sp
import os
import scipy.io as io
from helper import *
from result_struct import *


def get_Pearson_fc(sub_region_series,threshold,numnodes=90):
    subj_fc_adj = np.corrcoef(np.transpose(sub_region_series))
    subj_fc_adj_up=subj_fc_adj[np.triu_indices(numnodes,k=1)]
    subj_fc_adj_list = subj_fc_adj_up.reshape((-1))
    thindex = int(threshold * subj_fc_adj_list.shape[0])
    thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex-1]]
    subj_fc_adj_t = np.zeros((numnodes, numnodes))
    subj_fc_adj_t[subj_fc_adj > thremax] = 1
    subj_fc_adj=subj_fc_adj_t
    return subj_fc_adj


def K_Fold(folds, dataset, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset)), torch.tensor([data.y for data in dataset])):
        test_indices.append(index)
    return test_indices


class FSDataset(object):
    def __init__(self, root_dir, folds, seed):

        if 'zhongdaxinxiang_new' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "*.pkl"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                subj_fc_mat=data.source_mat[:230,:]
                # subj_fc_feature = max_min_norm(subj_fc_mat)
                subj_fc_feature = subj_fc_mat
                feature = np.corrcoef(subj_fc_feature.T)
                adj=get_Pearson_fc(subj_fc_feature,0.2)
                
                # adj = data.adjacency_mat
                # feature = np.corrcoef(data.source_mat.T)
                # edge_weight = adj * feature
                # edge_weight = feature[adj==1]

                # edge_index = adj['corr_each_sub']
                # edge_index = np.nan_to_num(edge_index)
                # edge_index_temp = sp.coo_matrix(edge_index)
                # edge_weight = torch.Tensor(edge_index_temp.data)
                # edge_index = torch.Tensor(edge_index)
                # edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()

                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(subj_fc_feature).float(), edge_index=fcedge_index,
                    edge_weight=torch.from_numpy(feature).float(), y=torch.as_tensor([label]).long()
                ))
        elif 'MDD_wwh_667' in root_dir:
            data_files = glob(os.path.join(root_dir,"**","*.mat"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                subj_fc_mat=io.loadmat(file)['ROISignals_AAL'][:230,:]
                # subj_fc_feature = max_min_norm(subj_fc_mat)
                subj_fc_feature = subj_fc_mat
                feature = np.corrcoef(subj_fc_feature.T)
                adj=get_Pearson_fc(subj_fc_feature,0.2)
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(subj_fc_feature).float(), edge_index=fcedge_index, edge_weight=torch.from_numpy(feature).float(), y=torch.as_tensor([label]).long()
                ))

        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)


    def kfold_split(self, batch_size, test_index):
        assert test_index < self.k_fold
        # valid_index = (test_index + 1) % self.k_fold
        valid_index = test_index
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.fc))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.fc, train_split.tolist())
        valid_subset = Subset(self.fc, valid_split.tolist())
        test_subset = Subset(self.fc, test_split.tolist())
        return train_subset, valid_subset, test_subset

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.fc)



# myDataset = FSDataset('D:/Data/ABIDE_useful', 10, 1)