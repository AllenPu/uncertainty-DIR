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
from utils import AverageMeter, accuracy, shot_metric, setup_seed, balanced_metrics, shot_metric_balanced, uncertainty_accumulation, label_uncertainty_accumulation
import torch
from loss import *
from network import *
import torch.optim as optim
import time
from scipy.stats import gmean
from conform import *
import itertools


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
parser.add_argument('--loss', type=str, default='l1', choices=[
                    'mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--epoch', type=int, default=201,
                    help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--print_freq', type=int,
                    default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
#
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--la', action='store_true',
                    help='if use logit adj to train the imbalance')
parser.add_argument('--model_depth', type=int, default=18,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
parser.add_argument('--g_dis', action='store_true',
                    help='if dynamically adjust the tradeoff')
parser.add_argument('--gamma', type=float, default=5, help='tradeoff rate')
parser.add_argument('--reweight', type=str, default=None,
                    help='weight : inv or sqrt_inv')
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
parser.add_argument('--scale', type=float, default=1, help='scale of the sharpness in soft label')
#parser.add_argument('--diversity', type=float, default=0, help='scale of the diversity loss in regressor output')
parser.add_argument('--fd_ratio', type=float, default=0, help='scale of the diversity loss in z')
parser.add_argument('--asymm', action='store_true', help='if use the asymmetric soft label')
parser.add_argument('--weight_norm', action='store_true', help='if use the weight norm for train')
parser.add_argument('--feature_norm', action='store_true', help='if use the feature norm for train')
parser.add_argument('--beta', default=0.5, type=float,  help='beta for nll')
parser.add_argument('--MSE', action='store_true', help='only use  MSE or not')
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
                          split='train', group_num=args.groups)
    
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
    return train_loader, test_loader, val_loader, train_labels


def train_one_epoch(args, model, train_loader, cal_loader, opts):
    model.train()
    #
    tau_low, tau_high = args.tau/2, 1- args.tau/2
    #
    [opt_model] = opts
    #
    var_list, label_list, pred_list = [], [], []
    #
    infinite_cal_loader = itertools.cycle(cal_loader)
    #
    #for idx, (x, y, w) in enumerate(train_loader):
    for train_batch, cal_batch in zip(train_loader, infinite_cal_loader):
        #
        x, y, w = train_batch
        #print('shape is', x.shape, y.shape, g.shape)
        #
        x, y, w  = x.to(device), y.to(device), w.to(device)
        #
        y_pred, lower, upper, z = model(x)
        #
        mse = F.mse_loss(y_pred, y, reduction='mean')
        #
        upper_loss = pinball_loss(y, upper, tau=tau_high)
        lower_loss = pinball_loss(y, lower, tau=tau_low)
        #
        #
        interval = abs_err(model, cal_batch, tau=0.1)
        interval = interval.expand_as(y)
        #
        if args.MSE:
            nll_loss = mse
        else:
            nll_loss = beta_nll_loss(y_pred, interval, y, args.beta)
            nll_loss *= w.expand_as(nll_loss)
            nll_loss = torch.mean(nll_loss)
        #
        #variance_loss = F.mse_loss(var_pred, var.to(torch.float32))
        loss = nll_loss.to(torch.float) + upper_loss + lower_loss
        #loss = loss.to(torch.float)
        #
        opt_model.zero_grad()
        loss.backward()
        opt_model.step()
        #    
        #label_list.append(y)
        #pred_list.append(y_pred)
    print(f' Nll is {nll_loss.item()} MSE is {mse.item()}')
    #
    '''
    vars, labels, preds  = torch.cat(var_list, 0), torch.cat(label_list, 0), torch.cat(pred_list, 0)
    if args.MSE:
        # the variance from the model output
        uncer_maj, uncer_med, uncer_low, uncer_total = 0, 0, 0, 0
        # the variance from the target predictions
        uncer_pred_maj, uncer_pred_med, uncer_pred_low, uncer_pred_total = \
            label_uncertainty_accumulation(preds, labels, maj, med, low, device)
    else:
        # the variance from the model output
        uncer_maj, uncer_med, uncer_low, uncer_total  = \
            uncertainty_accumulation(vars, labels, maj, med, low, device)
        # the variance from the target predictions
        uncer_pred_maj, uncer_pred_med, uncer_pred_low, uncer_pred_total  = \
            label_uncertainty_accumulation(preds, labels, maj, med, low, device)
    #
    results = [str(uncer_maj), str(uncer_med), str(uncer_low), str(uncer_total), str(nll_loss.item()), str(mse.item())]
    #
    vars_results_from_pred = [str(uncer_pred_maj), str(uncer_pred_med), str(uncer_pred_low), str(uncer_pred_total)]
    #
    #print(f' maj uncertainty {uncer_maj} med uncertainty {uncer_med} low uncertainty {uncer_low} total uncertainty {uncer_total}')
    #print(f' nll loss is {nll_loss}')
    #print(f' MSe is {mse}')
    '''
    return model#, results, vars_results_from_pred


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
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            bsz = x.shape[0]
            x, y= x.to(device), y.to(device)
            #
            labels.extend(y.data.cpu().numpy())
            #
            y_pred, lower, upper, z  = model(x)
            #
            #print(f' y shape is  {y_output.shape}')
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
        # gmean
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_pred = shot_metric(pred, labels, train_labels)
    print(f' MSE is {mse_pred.avg}')

    return mae_pred.avg, shot_pred, gmean_pred
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


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    store_name = ''
    #
    train_loader, test_loader, val_loader,  train_labels = get_data_loader(args)
    #
    loss_mse = nn.MSELoss()
    #
    maj, med, low = shot_count(train_labels)
    #
    model = ResNet_conformal(args).to(device)
    #
    #feature_dim = model.feature_dim
    #
    #mi_estimator = KNIFE(args, feature_dim).to(device)
    #
    opt_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #opt_mi = optim.Adam(mi_estimator.parameters(), lr=0.001, betas=(0.5, 0.999))
    #
    opts = [opt_model]#, opt_mi] 
    #
    output_file = '_beta_' + str(args.beta) +  'MSE' + str(args.MSE) + '.txt'
    #output_file = 'nll_output_vs_pred' + '_beta_' + str(args.beta) + '.txt'
    #
    for e in tqdm(range(args.epoch)):
        model = train_one_epoch(args, model, train_loader, val_loader, opts)
        #
        # record the prediction variance (from predicted labels) and model output variance respectively
        #
        '''
        with open('tr_output_variance' + output_file, "a+") as file:
            file.write(str(e)+" ")
            file.write(" ".join(results) + '\n')
            #file.write(" ".join(pred_results) + '\n')
            file.close()
        with open('tr_pred_var' + output_file, "a+") as file:
            file.write(str(e)+" ")
            file.write(" ".join(pred_results) + '\n')
            file.close()
        #
        '''
        if e%20 == 0:
    # test final model
            mae_pred, shot_pred, gmean_pred  = test(model, test_loader, train_labels, args)
    #
            print('=---------------------------------------------------------------------=\n')
            print(f' store name is {store_name}')
        #
            print(' Prediction ALL MAE {} Many: MAE {} Median: MAE {} Low: MAE {}'.format(mae_pred, shot_pred['many']['l1'],
                                                                             shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
        #
            print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                         shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n")     
            print('---------------------------------------------------------------------\n')
    #
    #write_log('./output/'+store_name, mae_pred, shot_pred, gmean_pred)
    

################
#
# draw the beta-NLL variance with different variance
#
################