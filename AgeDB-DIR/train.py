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
from utils import AverageMeter, shot_count, shot_metric, setup_seed, per_label_var, per_label_mae, per_label_frobenius_norm, label_uncertainty_accumulation, uncertainty_accumulation
import torch
from loss import *
from network import *
import torch.optim as optim
import time
from scipy.stats import gmean
from split_CP import coverage_loss, calibrate_qhat_from_batch, calibrate_qhat_splitCP, cqr_pinball
import torch.nn.functional as F
import itertools


# current sota 7.73, 7.46, 7.76, 10.08
# g 10 lr 0.0002 epoch 450 sigma 2 temp 0.02

import os
os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training/optimization related
parser.add_argument('--seed', default=42)
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/root/autodl-tmp/data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint',
                    help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='',
                    help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--warmup_epoch', default=50, type=int, help='warm-up epochs')
parser.add_argument('--epoch', type=int, default=101,
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
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
parser.add_argument('--g_dis', action='store_true',
                    help='if dynamically adjust the tradeoff')
parser.add_argument('--gamma', type=float, default=5, help='tradeoff rate')
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
parser.add_argument('--beta', default=0.5, type=float,  help='beta for nll')
parser.add_argument('--lamb', default=0.9, type=float,  help='lamb for coverage')
parser.add_argument('--weight', default=1, type=float,  help='weight for cp_loss in total loss')
parser.add_argument('--alpha', default=0.1, type=float,  help='miscoverage level for conformal calibration')
parser.add_argument('--cp_mode', type=str, default='hybrid', choices=['cqr', 'split', 'hybrid'])
#
parser.add_argument('--asymm', action='store_true', help='if use the asymmetric soft label')
parser.add_argument('--weight_norm', action='store_true', help='if use the weight norm for train')
parser.add_argument('--feature_norm', action='store_true', help='if use the feature norm for train')
#
# MSE only, else NLL
parser.add_argument('--MSE', action='store_true', help='only use MSE or not')
parser.add_argument('--MAE', action='store_true', help='only use MAE or not')
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
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_labels


def resolve_stage_mode(args, epoch):
    if args.cp_mode == 'hybrid':
        return 'split' if epoch < args.warmup_epoch else 'cqr'
    return args.cp_mode


def reduce_batch_loss(loss, weight, smooth_mode):
    if smooth_mode == 'lds':
        return (loss * weight.expand_as(loss)).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


def train_one_epoch(args, model, train_loader, cal_loader, opts, epoch):
    stage_mode = resolve_stage_mode(args, epoch)
    model.train()
    cal_iter = itertools.cycle(cal_loader)
    #
    opt_extractor, opt_regressor, opt_cp_upper, opt_cp_lower = opts
    #
    interval_list, label_list, pred_list, z_list = [], [], [], []
    nll_loss_history = []
    #
    for idx, (x, y, w) in enumerate(train_loader):
        #print('shape is', x.shape, y.shape, g.shape)
        #
        x, y, w  = x.to(device), y.to(device), w.to(device)
        if stage_mode == 'split':
            cal_batch = next(cal_iter)
            y_pred, _, _, z = model(x)
            q_hat = calibrate_qhat_splitCP(model, cal_batch, device, alpha=args.alpha)
            interval = torch.clamp(torch.ones_like(y_pred) * (2 * q_hat), min=1e-6)
            variance_for_nll = interval.detach().clamp_min(1e-6)

            if args.MSE:
                nll_loss = (y_pred - y) ** 2
            elif args.MAE:
                nll_loss = torch.abs(y_pred - y)
            else:
                nll_loss = beta_nll_loss(y_pred, variance_for_nll, y, args.beta)
            nll_loss = reduce_batch_loss(nll_loss, w, args.smooth) #lds

            opt_extractor.zero_grad()
            opt_regressor.zero_grad()
            nll_loss.backward()
            opt_extractor.step()
            opt_regressor.step()

        else:
            cal_batch = next(cal_iter)
            y_pred, _, _, z = model(x)

            # Keep CQR losses on the interval heads while the backbone continues
            # to train with the main regression objective in a separate step.
            z_det = z.detach()
            lower_cp = model.interval_lower(z_det)
            upper_cp = model.interval_upper(z_det)
            quantile_coverage = 1.0 - args.alpha
            loss_lower, loss_upper = cqr_pinball(y, upper_cp, lower_cp, quantile_coverage)

            x_cal, y_cal, _ = cal_batch
            x_cal, y_cal = x_cal.to(device), y_cal.to(device)
            with torch.no_grad():
                y_cal_pred, _, _, z_cal = model(x_cal)
            z_cal_det = z_cal.detach()
            lower_cal = model.interval_lower(z_cal_det)
            upper_cal = model.interval_upper(z_cal_det)
            cp_loss = coverage_loss(y_cal, y_cal_pred.detach(), lower_cal, upper_cal, lamb=args.lamb)
            interval_head_loss = loss_lower + loss_upper + args.weight * cp_loss

            opt_cp_lower.zero_grad()
            opt_cp_upper.zero_grad()
            interval_head_loss.backward()
            opt_cp_upper.step()
            opt_cp_lower.step()

            q_hat = calibrate_qhat_from_batch(model, cal_batch, device, alpha=args.alpha)
            y_pred, lower, upper, z = model(x)
            interval = torch.clamp(torch.abs(upper - lower) + 2 * q_hat, min=1e-6)
            variance_for_nll = interval.detach().clamp_min(1e-6)

            if args.MSE:
                nll_loss = (y_pred - y) ** 2
            elif args.MAE:
                nll_loss = torch.abs(y_pred - y)
            else:
                nll_loss = beta_nll_loss(y_pred, variance_for_nll, y, args.beta)
            nll_loss = reduce_batch_loss(nll_loss, w, args.smooth)

            #variance_loss = F.mse_loss(var_pred, var.to(torch.float32))
            loss = nll_loss.to(torch.float)#+ variance_loss
            opt_extractor.zero_grad()
            opt_regressor.zero_grad()
            loss.backward()
            opt_extractor.step()
            opt_regressor.step()

            #loss = loss.to(torch.float)
            #
        #
        interval_list.append(interval)
        label_list.append(y)
        pred_list.append(y_pred)
        z_list.append(z)
        nll_loss_history.append(nll_loss.item())
    #
    vars, labels, preds, z_  = torch.cat(interval_list, 0), torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
    epoch_nll_loss = float(np.mean(nll_loss_history)) if nll_loss_history else 0.0
    #
    #mae_dict = per_label_mae(preds , labels)
    #mae_dict = per_label_frobenius_norm(z_, labels)
    #
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
    pred_results = [str(uncer_maj), str(uncer_med), str(uncer_low), str(uncer_total), str(epoch_nll_loss)]
    #
    vars_results_from_pred = [str(uncer_pred_maj), str(uncer_pred_med), str(uncer_pred_low), str(uncer_pred_total)]
    #
    

    return model, pred_results,  vars_results_from_pred


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
            y_pred, lower, upper, z = model(x)
            interval = torch.clamp(torch.abs(upper - lower), min=1e-6)
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
            label_list.append(y)
            pred_list.append(y_pred)
            z_list.append(z)
        label_, pred_, z_  = torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
        #
        # gmean
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_pred = shot_metric(pred, labels, train_labels)
    print(f' MSE is {mse_pred.avg}')
    #
    #mae_dict = per_label_mae(pred_, label_)
    #mae_dict = per_label_frobenius_norm(z_, label_)
    #var_per_label = per_label_var(pred, labels)
    #mae_per_label = per_label_mae(pred_, label_)
    #
    #
    return mae_pred.avg#, shot_pred, gmean_pred, var_per_label, mae_per_label
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




if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    store_name = ''
    #
    train_loader, val_loader, test_loader,  train_labels = get_data_loader(args)
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
    opt_extractor = optim.Adam(model.model_extractor.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_regressor = optim.Adam(model.pred_head.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_cp_upper = optim.Adam(model.interval_upper.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_cp_lower = optim.Adam(model.interval_lower.parameters(), lr=args.lr, weight_decay=5e-4)
    opts = [opt_extractor, opt_regressor, opt_cp_upper, opt_cp_lower]
    #opt_mi = optim.Adam(mi_estimator.parameters(), lr=0.001, betas=(0.5, 0.999))
    #
    #opts = [opt_model]#, opt_mi#] 
    #
    output_file = 'beta_' + str(args.beta) + '.txt'
    #output_file = 'nll_output_vs_pred' + '_beta_' + str(args.beta) + '.txt'
    #
    for e in tqdm(range(args.epoch)):
        model, pred_results,  vars_results_from_pred = train_one_epoch(args, model, train_loader, val_loader, opts, e)
        mae_pred = test(model, test_loader, train_labels, args)
        #
        # record the prediction variance (from predicted labels) and model output variance respectively
        #
        
        with open('nll_' + output_file, "a+") as file:
            file.write(str(e)+" ")
            file.write(" ".join(pred_results) + " "+ " ".join(vars_results_from_pred) + " " + str(mae_pred) + '\n')
            #file.write(" ".join(vars_results_from_pred) + '\n')
            file.close()
        
        #
        #if e == 0 or e == args.epoch - 1:
        #    print(f'================Epoch is {e}================')
        #    _, _, _, _ = test(model, train_loader, train_labels, args)
        #    print('================End Cal================')
        '''
        if e % 1 == 1: #== args.epoch - 1:
            #assert 1 == 2
            # test final model
            #
            mae_pred, shot_pred, gmean_pred, mae_pred_te  = test(model, test_loader, train_labels, args)
            #
            print('=---------------------------------------------------------------------=\n')
            print(f' store name is {store_name} epoch is {e}')
            #
            print(' Prediction ALL MAE {} Many: MAE {} Median: MAE {} Low: MAE {}'.format(mae_pred, shot_pred['many']['l1'],
                                                                             shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
            #
            print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                         shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n")     
            print('---------------------------------------------------------------------\n')
            #
            mae_pred, _, _, _  = test(model, train_loader, train_labels, args)
            #
            print("----------train-----------")
            list_key_tr = [k for k in mae_pred_tr.keys()]
            print(list_key_tr)
            # print per-label MAE
            list_results_tr = [mae_pred_tr[k] for k in mae_pred_tr.keys()]
            #
            mae_pred, _, _, _  = test(model, train_loader, train_labels, args)
            print(f'Overall MAE for train is {mae_pred}')
            #
            print(list_results_tr)
            print("----------test-----------")
            list_key_te = [k for k in mae_pred_te.keys()]
            print(list_key_te)
            # print per-label MAE
            list_results_te = [mae_pred_te[k] for k in mae_pred_te.keys()]
            #
            print(list_results_te)
            #
            '''
    #write_log('./output/'+store_name, mae_pred, shot_pred, gmean_pred)
    

################
#
# draw the beta-NLL variance with different variance
#
################
