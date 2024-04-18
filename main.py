import argparse
import scipy
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
import os
import numpy as np
import torch.nn as nn

import torch

from SMFCNet_pytorch import SMFC_Net
from train_eval import train_model, eval_FCSC
from load_data import FSDataset
from helper import *


parser = argparse.ArgumentParser(description='classification for brain graph')
parser.add_argument('--input_dim', type=int, default=246, help='fc_input_feature_dim') #useless
# parser.add_argument('--num_nodes', type=int, default=90, help='num_nodes') #roi
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes') #useless
# parser.add_argument('--datapath', type=str, default='/home/weijiayin/bioblank/sc_bn_sex/sc_bn_sex', help='path of dataset')
# parser.add_argument('--savepath', type=str, default='/home/weijiayin/bioblank/biobank_code/biobank/ckpt', help='path of model')
# parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs')
# parser.add_argument('--folds', type=int, default=10, help='k-fold cross validation')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
# parser.add_argument('--device', type=str, default='cuda:0', help='cuda devices')

parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim') #useless
# parser.add_argument('--batch_size', type=int, default=512, help='batch size')
# parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--num_layers', type=int, default=2, help='the numbers of convolution layers') #useless
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
# parser.add_argument('--seed', type=int, default=123, help='random seed')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--num_nodes', type=int, default=90, help='num_nodes') #roi
parser.add_argument('--folds', type=int, default=10, help='k-fold cross validation')
parser.add_argument('--data_seed', type=int, default=[50, 75, 100, 125, 150, 175, 200, 225, 250, 275], help='data seed')  #每次的种子点
parser.add_argument('--device', type=str, default='cuda:0', help='cuda devices')
parser.add_argument('--repetitions', type=int, default=1, help='number of repetitions')

parser.add_argument('--datapath', type=str, default='data/MDD_wwh_667', help='path of dataset')
parser.add_argument('--savepath', type=str, default='result_wwh', help='path of model')

parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

# add 
parser.add_argument('--SBN_num', type=float, default=10, help='SBN_num subset')

args = parser.parse_args()



file = 'result_wwh/results.txt'


if __name__ == '__main__':
    acc = []
    sen = []
    spe = []
    f1 = []
    auc = []
    print(torch.cuda.is_available())

    setup_seed(args.seed)
    print(args)

    for k in range(args.repetitions):
        print('begin loading dataset')
        myDataset = FSDataset(args.datapath, args.folds, args.data_seed[k],args)
        print('end loading dataset')

        acc_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []
        aucprop_iter = []
        att_iter = []
        for i in range(args.folds):
            print('fold:',i)

            
            # train_loader, val_loader, test_loader = myDataset.kfold_split(args.batch_size, i)
            train_dataset, val_dataset, test_dataset = myDataset.kfold_split(args.batch_size, i)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            print('begin loading model')
            model = SMFC_Net(args).to(args.device)
            print('end loading model')

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100], gamma=0.5)
            print('begin train model')
            best_model = train_model(args, model, optimizer, scheduler, train_loader, val_loader, i)
            print('end train model')

            model.load_state_dict(torch.load('{}/{}_fold_best_model.pth'.format(args.savepath, i)))
            test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, y, pred = eval_FCSC(args, model, test_loader)
            acc_iter.append(test_acc*100)
            sen_iter.append(test_sen*100)
            spe_iter.append(test_spe*100)
            f1_iter.append(test_f1*100)
            auc_iter.append(test_auc*100)
            print('Test set results, best_epoch = {:.1f}, loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                  'specificity = {:.6f}, f1 = {:.6f}, auc = {:.6f}'.format(best_model, test_loss, test_acc, test_sen,
                                                                           test_spe, test_f1, test_auc))
            print('begin saving result',i)
            save_each_fold(file, test_acc, test_sen, test_spe, test_f1, test_auc)
            save_pred(file,list(np.array(y).reshape(-1)), pred)
            print('end saving result',i)


        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))
        f1.append(np.mean(f1_iter))
        auc.append(np.mean(auc_iter))
        print('begin saving all result,repetition',k)
        save_std(file, acc_iter, sen_iter, spe_iter, f1_iter, auc_iter)
        print('end saving all result,repetition',k)


    print(args)

    save_std(file, acc, sen, spe, f1, auc)
    save_args(file, args)
    print('process over!!!!')
