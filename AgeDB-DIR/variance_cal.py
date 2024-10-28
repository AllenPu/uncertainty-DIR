import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def count_down(labels, y_gt, y_pred, y_uncertain):
    gt_list, pred_list, uncertain_list = [], [], []
    for i in labels:
        #
        indexes = [i for i,x in enumerate(y_pred) if x==i]
        gt_indexes = [i for i,x in enumerate(y_gt) if x==i]
        #
        gt = [y_gt[i] for i in gt_indexes]
        preds = [y_pred[i] for i in indexes]
        uncertains = [y_uncertain[i] for i in indexes]
        #
        gt_list.append(torch.sum(torch.Tensor(gt)).data)
        uncertain_list.append(torch.mean(torch.Tensor(uncertains)).data)
        pred_list.append(torch.sum(torch.Tensor(preds)).data)
    return gt_list, uncertain_list, pred_list


# the prediction variance of the training
def variance_calculation(model, train_loader):
    y_gt, y_pred, y_uncertain = [], [], []
    for idx, (x, y, g) in enumerate(train_loader):
        with torch.no_grad():
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred, uncertain = model(x)
            sigma = torch.sqrt(torch.exp(torch.abs(uncertain)))
            y_gt.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_uncertain.extend(sigma.cpu().numpy())
        #
    labels = np.unique(y_gt).tolist()
    #
    y_gt, y_pred, y_uncertain = np.hstack(y_gt).tolist(), np.hstack(y_pred).tolist(), np.hstack(y_uncertain).tolist()
    #
    y_pred = [math.ceil(i) if (i-int(i))> 0.5 else math.floor(i) for i in y_pred]
    #
    gt_list, uncertain_list, pred_list = count_down(labels, y_gt, y_pred, y_uncertain)
    #
    plt.plot(labels, gt_list, 'r--', uncertain_list, 'bs',  pred_list,  'g^' )
    plt.legend()
    #plt.show()
    plt.savefig('./var_scatter.png')
    plt.clf()
    gt_data, uncertain_data, pred_data = np.array(gt_list), np.array(uncertain_list), np.array(pred_list)
    gt_data, uncertain_data, pred_data = \
        gt_data.reshape(gt_data.shape[0], 1), uncertain_data.reshape(uncertain_data.shape[0], 1), pred_data.reshape(pred_data.shape[0], 1)
    datas = np.concatenate((gt_data, uncertain_data, pred_data),axis=1) 
    # ground truth label, prediction label, and prediction variance label
    plt.hist(datas, bins=len(gt_list),edgecolor = 'w',color = ['c','r', 'b'],  label = ['gt','pred','pred_var'], stacked = False)
    ax = plt.gca() 
    plt.legend()
    #plt.show()
    plt.savefig('./var_hist.png')