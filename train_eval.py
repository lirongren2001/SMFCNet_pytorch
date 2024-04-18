import time
import torch
import torch.nn.functional as F
from helper import *


def train_model(args, model, optimizer, scheduler, train_loader, val_loader, i_fold):
    max_acc = 0
    patience = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            # print('data.shaoe',data.x.shape)
            # print('y.shaoe',data.y.shape)

            out = model(data)
            loss = F.cross_entropy(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.x.size(0) / args.num_nodes
        scheduler.step()

        test_acc, test_loss, _, _, _, _, _, _ = eval_FCSC(args, model, val_loader)

        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss / len(train_loader.dataset)),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))

        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), '{}/{}_fold_best_model.pth'.format(args.savepath, i_fold))
            print("Model saved at epoch{}".format(epoch))
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))
    return best_epoch


def eval_FCSC(args, model, loader):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    file_count = 0
    for data in loader:
        with torch.no_grad():
            file_count+=1
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            # print('out',out,out.shape)
            # print('pred',pred,pred.shape)
            # print('data.y.view(-1)',data.y.view(-1))
            # print('correct=',correct)
            test_loss += F.cross_entropy(out, data.y.view(-1), reduction='sum').item()
            for num in range(len(pred)):
                Y_pred.append(pred.cpu().numpy()[num])
                Y_test.append(data.y.cpu().numpy()[num])
    # print('add correct',correct)
    # print('file_count',file_count)
    # print('len(loader.dataset)',len(loader.dataset))
    test_acc = correct / len(loader.dataset)
    test_loss = test_loss / len(loader.dataset)
    # test_acc = correct / file_count
    # test_loss = test_loss / file_count
    test_sen, test_spe, test_f1, test_auc = sensitivity_specificity(Y_test, Y_pred)
    return test_acc, test_loss, test_sen, test_spe,test_f1,test_auc,Y_test, Y_pred
