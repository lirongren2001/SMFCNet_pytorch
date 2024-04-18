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
from helper import *
from result_struct import *
import re
import pandas as pd
from helper import *
import scipy.io as io

def threshold(data,n_subjects, n_regions, n_lambda):
    '''
    wsr的二值化。
    '''
    # 创建一个用于存储最终阈值后的数据的数组
    threshold_final = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    # 对每个lambda值进行循环
    for i in range(n_lambda):
        # 创建一个用于存储当前lambda值下的阈值后的数据的数组
        threshold = np.zeros((n_subjects, n_regions, n_regions))
        # 对每个受试者的脑网络数据进行循环
        for j in range(n_subjects):
            # 获取当前lambda值和受试者的脑网络数据
            sub_wsr = data[i][j]
            # 将非零元素转换为1，零元素转换为0，实现二值化
            _single_threshold = (sub_wsr != 0).astype(int)
            # 将阈值后的数据存储到数组中
            threshold[j,:] = _single_threshold
        # 将当前lambda值下的阈值后的数据存储到最终数组中
        threshold_final[i,:] = threshold
    return threshold_final # n_lambda, n_subjects, n_regions, n_regions

def K_Fold(folds, dataset, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset)), torch.tensor([data.y for data in dataset])):
        test_indices.append(index)
    return test_indices

def get_Pearson_fc(feature_matrix, threshold):
    pearson_corr = np.corrcoef(feature_matrix)
    adj_matrix = np.where(np.abs(pearson_corr) > threshold, 1, 0)

    return adj_matrix

class FSDataset(object):
    def __init__(self, root_dir, folds, seed, args,threshold=None, num_nodes=None, n_neighbors=None):
        file_count = 0
        if 'zhongdaxinxiang_new' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "*.pkl"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                file_count = file_count + 1
                with open(file, "rb") as f:
                    data = pickle.load(f)
                adj = data.adjacency_mat
                feature = np.corrcoef(data.source_mat.T)
                
                # feature = data.source_mat[:230,:]
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor([label]).long()
                ))
            # 对WSR数据进行阈值化处理
            # self.fc.x = threshold(self.fc.x,file_count,args.num_nodes,args.num_nodes)
        elif 'MDD_wwh_667' in root_dir:
            data_files = glob(os.path.join(root_dir,"**","*.mat"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                subj_fc_mat=io.loadmat(file)['ROISignals_AAL'][:230,:]
                subj_fc_feature=max_min_norm(subj_fc_mat)
                adj=get_Pearson_fc(subj_fc_feature,0.2)
                feature = np.corrcoef(subj_fc_feature.T)
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(subj_fc_feature.T).float(), edge_index=fcedge_index, y=torch.as_tensor([label]).long()
                ))

        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)
        # print('self.fc',self.fc.shape)


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

