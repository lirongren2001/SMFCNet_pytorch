import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from load_data import load_SMFCN_label
from SMFCNet import SMFC_Net

ex_num = 10
ex_fold_num = 5
in_num = 5
in_fold_num = 5
learning_rate = 1e-3
batch_size = 20
epochs = 200
test_num = 25
SBN_num = 10

data_all, label_all = load_SMFCN_label()

all_mean_result_ex = np.array(())
all_result_ex = np.zeros((ex_num, ex_fold_num))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for jj in range(ex_num):
    np.random.seed(jj)
    torch.manual_seed(jj)
    kf = StratifiedKFold(n_splits=ex_fold_num, shuffle=True, random_state=jj)
    h = 0
    everyresult_ex = np.array(())

    for train_index, test_index in kf.split(data_all, label_all):
        h += 1
        train_X, train_y = data_all[train_index], label_all[train_index]
        test_X, test_y = data_all[test_index], label_all[test_index]
        num_train_data_ex = train_X.shape[0]

        train_X = torch.from_numpy(train_X).float().to(device)
        train_y = torch.from_numpy(train_y).long().to(device)
        test_X = torch.from_numpy(test_X).float().to(device)
        test_y = torch.from_numpy(test_y).long().to(device)

        all_mean_result = np.array(())
        in_all_result = np.zeros((in_num, in_fold_num))

        for pp in range(in_num):
            kf_in = StratifiedKFold(n_splits=in_fold_num, shuffle=True, random_state=pp)
            everyresult = np.array(())
            m = 0

            for train_index_in, dev_index in kf_in.split(train_X, train_y):
                m += 1
                train_X_in, train_y_in = train_X[train_index_in], train_y[train_index_in]
                dev_X, dev_y = train_X[dev_index], train_y[dev_index]
                num_train_data_in = train_X_in.shape[0]

                model = SMFC_Net().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()

                inputs_list = [train_X_in[:, i, :] for i in range(SBN_num)]
                inputs_list_dev = [dev_X[:, i, :] for i in range(SBN_num)]

                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(inputs_list)
                    loss = criterion(outputs, train_y_in)
                    loss.backward()
                    optimizer.step()

                model.eval()
                train_acc = torch.sum(torch.argmax(model(inputs_list), dim=1) == train_y_in).item() / num_train_data_in
                dev_acc = torch.sum(torch.argmax(model(inputs_list_dev), dim=1) == dev_y).item() / len(dev_y)
                print('Train: {:.3f}, Test: {:.3f}'.format(train_acc, dev_acc))

                torch.save(model.state_dict(), f'trained_model/model_exnum_{jj}_exfold_{h}_innum_{pp}_infold_{m}.pt')

                with torch.no_grad():
                    model.eval()
                    dev_outputs = model(inputs_list_dev)
                    _, predicted = torch.max(dev_outputs, 1)
                    acc = (predicted == dev_y).sum().item() / len(dev_y)
                    everyresult = np.append(everyresult, acc)

            in_all_result[pp, :] = everyresult
            mean_acc_in = np.sum(everyresult) / in_fold_num
            all_mean_result = np.append(all_mean_result, mean_acc_in)

        acc_final = np.sum(all_mean_result) / in_num

        inputs_list_test = [test_X[:, i, :] for i in range(SBN_num)]

        all_ex_result = ()

        for innum in range(in_num):
            every_fold_ex = np.zeros((in_fold_num, test_num))

            for infold in range(in_fold_num):
                model = SMFC_Net().to(device)
                model.load_state_dict(torch.load(f'trained_model/model_exnum_{jj}_exfold_{h}_innum_{innum}_infold_{infold+1}.pt'))
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs_list_test)
                    _, predicted = torch.max(outputs, 1)
                    every_fold_ex[infold, :] = predicted.cpu().numpy()

            all_ex_result += (every_fold_ex,)

        all_ex_result_y_pred = np.vstack(all_ex_result)

        y_pred_fusion = np.sum(all_ex_result_y_pred, axis=0)
        y_pred_final = np.array(())
        y_fusion_prob = np.zeros((test_num, 2))

        for bb in range(test_num):
            one_prob = y_pred_fusion[bb] / classifiers
            zero_prob = 1 - (y_pred_fusion[bb] / classifiers)
            prob = np.array((zero_prob, one_prob))
            y_fusion_prob[bb, :] = prob

            if (y_pred_fusion[bb] / classifiers) >= 0.5:
                y_pred_final = np.append(y_pred_final, 1)
            else:
                y_pred_final = np.append(y_pred_final, 0)

        y_true_ex = test_y.cpu().numpy()

        final_acc = np.sum(y_pred_final == y_true_ex) / test_num
        print("each_fold_acc_ex: {:.7f}".format(final_acc))
        everyresult_ex = np.append(everyresult_ex, final_acc)

    all_result_ex[jj, :] = everyresult_ex
    mean_acc_ex = np.sum(everyresult_ex) / ex_fold_num
    all_mean_result_ex = np.append(all_mean_result_ex, mean_acc_ex)
    print("mean_acc_ex: {:.7f}".format(mean_acc_ex))

acc_final_ex = np.sum(all_mean_result_ex) / ex_num
print("acc_final_ex: {:.7f}".format(acc_final_ex))
