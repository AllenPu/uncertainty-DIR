import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from agedb import *
from utils import AverageMeter, shot_metric, setup_seed, per_label_var, per_label_mae, per_label_frobenius_norm
import torch
from loss import *
from network import *
import torch.optim as optim
import time
from scipy.stats import gmean
from distloss import DistLoss,get_label_distribution, get_batch_theoretical_labels
import itertools
from conform_cqr import *
from conform_label_shift import *




# current sota 7.73, 7.46, 7.76, 10.08
# g 10 lr 0.0002 epoch 450 sigma 2 temp 0.02

import os
os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training/optimization related
parser.add_argument('--seed', default=3407)
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint',
                    help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='',
                    help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--epoch', type=int, default=101,
                    help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
#
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
#
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
#
parser.add_argument('--tau', default=1, type=float,
                    help=' tau for logit adjustment ')
parser.add_argument('--ranked_contra', action='store_true')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)
#
parser.add_argument('--soft_label', action='store_true')
parser.add_argument('--ce', action='store_true',  help='if use the cross_entropy /la or not')
parser.add_argument('--output_file', type=str, default='result_')
#parser.add_argument('--diversity', type=float, default=0, help='scale of the diversity loss in regressor output')
parser.add_argument('--fd_ratio', type=float, default=0, help='scale of the diversity loss in z')
#
parser.add_argument('--weight_norm', action='store_true', help='if use the weight norm for train')
parser.add_argument('--feature_norm', action='store_true', help='if use the feature norm for train')
# which loss
parser.add_argument('--diff_loss', action='store_true', help='use different loss for Maj or Min or not')
parser.add_argument('--dist_loss', action='store_true', help='use dist loss or not')
# first reweight and then judge if we can use LDS
parser.add_argument('--reweight', type=str, default='inv',  choices=['inv', 'sqrt_inverse'],
                    help='weight : inv or sqrt_inv')
parser.add_argument('--smooth', default='none', choices=['lds', 'none'], help='use LDS or not')
#
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def get_data_loader(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                          split='train', reweight=args.reweight, group_num=args.groups, smooth=args.smooth)   
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                         img_size=args.img_size, split='test', group_num=args.groups)
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    train_num_dict, train_weight_dict = train_dataset.get_weight_dict()
    return train_loader, val_loader, test_loader, train_labels, train_num_dict, train_weight_dict


def train_one_epoch(args, model, train_loader, cal_loader, opts):
    model.train()
    #
    [opt_model] = opts
    #
    var_list, label_list, pred_list, z_list = [], [], [], []
    #
    infinite_cal_loader = itertools.cycle(cal_loader)
    #
    for train_batch, cal_batch in zip(train_loader, infinite_cal_loader):
        #
        loss = 0
        #print('shape is', x.shape, y.shape, g.shape)
        x, y, w = train_batch
        x_cal, y_cal, _ = cal_batch
        #
        x, y, w, x_cal, y_cal  = x.to(device), y.to(device), w.to(device), x_cal.to(device), y_cal.to(device)
        #
        z, y_pred, var_pred = model(x)
        #
        #mse = F.mse_loss(y_pred, y, reduction='sum')
        if args.diff_loss:
            loss = train_with_different_loss(y, y_pred)
        elif args.dist_loss:
            loss = train_with_dist_loss(y, y_pred, theoretical_labels, dist_loss)
        else:
            # only MSE loss
            loss = torch.nn.functional.mse_loss(y_pred, y)
        #
        # label shift conformal regression 
        nll_loss = train_with_nll(x, y, y_pred, x_cal, y_cal)
        #
        loss += nll_loss
        #
        opt_model.zero_grad()
        loss.backward()
        opt_model.step()
        #
        var_list.append(var_pred)
        label_list.append(y)
        pred_list.append(y_pred)
        z_list.append(z)
    #
    vars, labels, preds, z_  = torch.cat(var_list, 0), torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
    #
    mae_dict = per_label_mae(preds , labels)
    #mae_dict = per_label_frobenius_norm(z_, labels)

    return model, mae_dict#vars_results_from_pred


def test(model, test_loader, train_labels, args):
    model.eval()
    #
    mse_pred = AverageMeter()
    mae_pred = AverageMeter()
    # gmean
    criterion_gmean_pred = nn.L1Loss(reduction='none')
    gmean_loss_all_pred = []
    #
    pred, labels = [], []
    #
    pred_list, label_list, z_list = [], [], []
    #
    with torch.no_grad():
        for idx, (x, y, _) in enumerate(test_loader):
            bsz = x.shape[0]
            x, y= x.to(device), y.to(device)
            #
            labels.extend(y.data.cpu().numpy())
            #
            z, y_pred, var_pred = model(x)
            #
            mae_y = torch.mean(torch.abs(y_pred- y))
            mse_y_pred = F.mse_loss(y_pred, y)
            #
            pred.extend(y_pred.data.cpu().numpy())
            # gmean
            loss_all_pred = criterion_gmean_pred(y_pred, y)
            gmean_loss_all_pred.extend(loss_all_pred.cpu().numpy())
            #
            mse_pred.update(mse_y_pred.item(), bsz)
            #
            mae_pred.update(mae_y.item(), bsz)
            #
            label_list.append(y)
            pred_list.append(y_pred)
            z_list.append(z)
        #
        label_, pred_, z_  = torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
        #
        # gmean
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_pred = shot_metric(pred, labels, train_labels)
    print(f' MSE is {mse_pred.avg}')
    #
    #mae_dict = per_label_mae(pred_, label_)
    '''
    mae_dict = per_label_frobenius_norm(z_, label_)
    var_per_label = per_label_var(pred, labels)
    mae_per_label = per_label_mae(pred_, label_)
    '''
    #
    #
    return mae_pred.avg, shot_pred, gmean_pred#, mae_dict
        # np.hstack(group), np.hstack(group_pred) #newly added





######################
# write log for the test
#####################
def write_log(store_name, mae_pred, shot_pred, gmean_pred):
    with open(store_name, 'a+') as f:
        f.write('=---------------------------------------------------------------------=\n')
        f.write(f' store name is {store_name}')
        #
        f.write(' Prediction ALL MAE {} Many: MAE {} Median: MAE {} Low: MAE {}'.format(mae_pred, shot_pred['many']['l1'],
                                                                             shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
        #
        f.write(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                         shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n")     
        f.write('---------------------------------------------------------------------\n')
        f.close()
#############################
# print the mae result of per label from low to high
def print_mae(mae_dict):
    all_labels, all_mae = [], []
    for k in sorted(mae_dict.keys()):
        all_labels.append(k)
        all_mae.append(mae_dict[k])
    print("-----all labels per mae-----")
    print(all_labels)
    print("-----all mae-----")
    print(all_mae)




def dist_loss_fn(train_labels, bw_method=0.5, min_label=0, max_label=120, step=1):
    density = get_label_distribution(train_labels, bw_method, min_label, max_label, step)
    batch_theoretical_labels = get_batch_theoretical_labels(density, batch_size=256, min_label=min_label, step=step)
    batch_theoretical_labels = torch.tensor(batch_theoretical_labels, dtype=torch.float32).reshape(-1,1).cuda()
    loss_fn = DistLoss()
    return batch_theoretical_labels, loss_fn



def train_with_dist_loss(y, y_pred, batch_theoretical_labels, loss_fn):
    loss = loss_fn(y_pred.type(torch.double), y.type(torch.double), batch_theoretical_labels.type(torch.double))
    return loss


def train_with_nll(x, y, y_pred, x_cal, y_cal):
    # start for the intervals
    #
    #  use "label shift conformal regression"
    interval = cal_interval(model, x_cal, y_cal, y_pred, reverse_train_dict)
    interval = interval.expand_as(y)
    nll_loss = beta_nll_loss(y_pred, interval, y, beta=0.5)
    return nll_loss


# use MSE for majority while MAE for minority
def train_with_different_loss(y, y_pred):
    maj_loss, med_loss, low_loss = 0, 0, 0
    y_ = y.view(-1)
    maj_mask = torch.isin(y_, maj)
    med_mask = torch.isin(y_, med)
    low_mask = torch.isin(y_, low)
    maj_indices = torch.nonzero(maj_mask).squeeze()
    med_indices = torch.nonzero(med_mask).squeeze()
    low_indices = torch.nonzero(low_mask).squeeze()
    #
    #print(f' y shape is  {y_output.shape}')
    #
    if maj_indices.numel() > 0:
        maj_loss = torch.mean(torch.abs(y_pred[maj_indices] - y[maj_indices]))
    if med_indices.numel() > 0:
        med_loss = torch.mean(torch.abs(y_pred[med_indices] - y[med_indices])**2)
    if low_indices.numel() > 0:
        low_loss = torch.mean(torch.abs(y_pred[low_indices] - y[low_indices])**2)
    #
    loss = maj_loss + med_loss + low_loss
    #
    return loss




if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    store_name = ''
    #
    train_loader, test_loader, val_loader,  train_labels, train_num_dict, train_weight_dict = get_data_loader(args)
    #
    loss_mse = nn.MSELoss()
    #
    maj, med, low = shot_count(train_labels)
    maj, med, low = torch.tensor(maj).to(device), torch.tensor(med).to(device), torch.tensor(low).to(device)
    reverse_train_dict = {}
    for k in train_num_dict.keys():
        reverse_train_dict[k] = 1/train_num_dict[k]
    #
    model = Guassian_uncertain_ResNet(name = 'resnet18', norm = args.feature_norm, weight_norm = args.weight_norm).to(device)
    #
    opt_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #opt_mi = optim.Adam(mi_estimator.parameters(), lr=0.001, betas=(0.5, 0.999))
    #
    opts = [opt_model]#, opt_mi#] 
    #
    output_file = 'dual_loss' + '.txt'
    #output_file = 'nll_output_vs_pred' + '_beta_' + str(args.beta) + '.txt'
    #
    if args.dist_loss:
        theoretical_labels, dist_loss = dist_loss_fn(train_labels=train_labels)
    #
    for e in tqdm(range(args.epoch)):
        model, mae_dict = train_one_epoch(args, model, train_loader, val_loader, opts)
        #
        # record the prediction variance (from predicted labels) and model output variance respectively 
        #
        if e == args.epoch - 1:
            #
            print_mae(mae_dict)
            #
            # test final model
            mae_pred, shot_pred, gmean_pred = test(model, test_loader, train_labels, args)
            #
            print('=---------------------------------------------------------------------=\n')
            print(f' Store name is {store_name} epoch is {e}')
            #
            print(' Prediction ALL MAE {} Many: MAE {} Median: MAE {} Low: MAE {}'.format(mae_pred, shot_pred['many']['l1'],
                                                                             shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
            #
            print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                         shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n")     
            print('---------------------------------------------------------------------\n')
            #
            mae_pred, _, _  = test(model, train_loader, train_labels, args)

