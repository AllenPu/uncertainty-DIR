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
from split_CP import coverage_loss, calibrate_qhat_from_batch, calibrate_qhat_splitCP, cqr_pinball, interval_minimization
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
parser.add_argument('--variance_mse_threshold', type=float, default=1,
                    help='after warmup, switch samples with variance below this threshold from NLL to MSE')
parser.add_argument('--lamb', default=0.9, type=float,  help='lamb for coverage')
parser.add_argument('--weight', default=1, type=float,  help='weight for cp_loss in total loss')
parser.add_argument('--alpha', default=0.1, type=float,  help='miscoverage level for conformal calibration')
parser.add_argument('--cp_mode', type=str, default='hybrid', choices=['cqr', 'split', 'hybrid'])
parser.add_argument('--warmup_ckpt_path', type=str, default='',
                    help='path to save the final warmup checkpoint; defaults to <store_root>/<store_name>/warmup_final.pth.tar')
parser.add_argument('--resume_warmup_ckpt', type=str, default='',
                    help='path to a saved warmup checkpoint to resume from')
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
parser.add_argument('--inv_method', default='cqr_pinball', choices=['split_cp', 'cqr_pinball', 'cqr_coverage'], help='use which method to train interval module')
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
        return 'warmup' if epoch < args.warmup_epoch else 'train'
    return args.cp_mode


def get_checkpoint_dir(args):
    return os.path.join(args.store_root, args.store_name) if args.store_name else args.store_root


def get_warmup_checkpoint_path(args):
    if args.warmup_ckpt_path:
        return args.warmup_ckpt_path
    return os.path.join(get_checkpoint_dir(args), 'warmup_final.pth.tar')


def save_warmup_checkpoint(args, model, opts, epoch):
    if args.warmup_epoch <= 0 or epoch != args.warmup_epoch - 1:
        return

    opt_extractor, opt_regressor, opt_cp_upper, opt_cp_lower = opts
    ckpt_path = get_warmup_checkpoint_path(args)
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'next_epoch': epoch + 1,
        'warmup_epoch': args.warmup_epoch,
        'model_state_dict': model.state_dict(),
        'opt_extractor_state_dict': opt_extractor.state_dict(),
        'opt_regressor_state_dict': opt_regressor.state_dict(),
        'opt_cp_upper_state_dict': opt_cp_upper.state_dict(),
        'opt_cp_lower_state_dict': opt_cp_lower.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f' Saved warmup checkpoint to {ckpt_path}')


def maybe_resume_from_warmup_checkpoint(args, model, opts):
    if not args.resume_warmup_ckpt:
        return 0

    ckpt_path = args.resume_warmup_ckpt
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    opt_extractor, opt_regressor, opt_cp_upper, opt_cp_lower = opts
    if 'opt_extractor_state_dict' in checkpoint:
        opt_extractor.load_state_dict(checkpoint['opt_extractor_state_dict'])
    if 'opt_regressor_state_dict' in checkpoint:
        opt_regressor.load_state_dict(checkpoint['opt_regressor_state_dict'])
    if 'opt_cp_upper_state_dict' in checkpoint:
        opt_cp_upper.load_state_dict(checkpoint['opt_cp_upper_state_dict'])
    if 'opt_cp_lower_state_dict' in checkpoint:
        opt_cp_lower.load_state_dict(checkpoint['opt_cp_lower_state_dict'])

    start_epoch = checkpoint.get('next_epoch', checkpoint.get('epoch', -1) + 1)
    print(f' Loaded warmup checkpoint from {ckpt_path}, resuming at epoch {start_epoch}')
    return start_epoch


def reduce_batch_loss(loss, weight, smooth_mode):
    if smooth_mode == 'lds':
        return (loss * weight.expand_as(loss)).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


def compute_interval_coverage(lower, upper, label, maj, med, low, device):
    covered = ((label >= lower) & (label <= upper)).to(torch.float)

    def group_coverage(group_labels):
        group_tensor = torch.as_tensor(group_labels, device=device)
        group_indices = torch.nonzero(torch.isin(label, group_tensor), as_tuple=False)
        if group_indices.numel() == 0:
            return float('nan')
        return covered[group_indices[:, 0]].squeeze(-1).mean().item()

    maj_cov = group_coverage(maj)
    med_cov = group_coverage(med)
    low_cov = group_coverage(low)
    total_cov = covered.squeeze(-1).mean().item()
    return maj_cov, med_cov, low_cov, total_cov


def maybe_switch_low_variance_to_mse(args, point_loss, mse_component, var_component, y_pred, interval, y):
    threshold = args.variance_mse_threshold
    if args.MSE or args.MAE or threshold is None:
        return point_loss, mse_component, var_component

    mse_loss = (y_pred - y) ** 2
    low_variance_mask = interval < threshold
    point_loss = torch.where(low_variance_mask, mse_loss, point_loss)
    mse_component = torch.where(low_variance_mask, mse_loss, mse_component)
    var_component = torch.where(low_variance_mask, torch.zeros_like(var_component), var_component)
    return point_loss, mse_component, var_component


def compute_point_loss_components(args, y_pred, interval, y, w):
    """Return total point loss plus beta-NLL subcomponents after batch reduction."""
    if args.MSE:
        point_loss = (y_pred - y) ** 2
        mse_component = point_loss
        var_component = torch.zeros_like(point_loss)
    elif args.MAE:
        point_loss = torch.abs(y_pred - y)
        mse_component = torch.zeros_like(point_loss)
        var_component = torch.zeros_like(point_loss)
    else:
        point_loss, mse_component, var_component = beta_nll_components(
            y_pred, interval, y, beta=args.beta
        )
        point_loss, mse_component, var_component = maybe_switch_low_variance_to_mse(
            args, point_loss, mse_component, var_component, y_pred, interval, y
        )

    point_loss = reduce_batch_loss(point_loss, w, args.smooth)
    mse_component = reduce_batch_loss(mse_component, w, args.smooth)
    var_component = reduce_batch_loss(var_component, w, args.smooth)
    return point_loss, mse_component, var_component

def train_one_epoch(args, model, train_loader, cal_loader, opts, epoch):
    stage_mode = resolve_stage_mode(args, epoch)
    model.train()
    cal_iter = itertools.cycle(cal_loader)
    #
    opt_extractor, opt_regressor, opt_cp_upper, opt_cp_lower = opts
    #
    interval_list, label_list, pred_list, z_list = [], [], [], []
    lower_list, upper_list = [], []
    nll_loss_history = []
    nll_mse_history = []
    nll_var_history = []
    #
    for idx, (x, y, w) in enumerate(train_loader):
        #print('shape is', x.shape, y.shape, g.shape)
        #
        x, y, w  = x.to(device), y.to(device), w.to(device)
        train_batch = (x, y, w)

        if stage_mode == 'warmup':
            y_pred, _, _, z = model(x)
            interval = torch.zeros_like(y_pred)
            mse_loss = (y_pred - y) ** 2
            nll_loss = reduce_batch_loss(mse_loss, w, args.smooth)
            nll_mse_component = nll_loss.detach()
            nll_var_component = torch.zeros_like(nll_loss)
            lower, upper = None, None

            opt_extractor.zero_grad()
            opt_regressor.zero_grad()
            nll_loss.backward()
            opt_extractor.step()
            opt_regressor.step()

        else:
            #
            cal_batch = next(cal_iter)
            x_cal, y_cal, _ = cal_batch
            x_cal, y_cal = x_cal.to(device), y_cal.to(device)
            # from warm_up to train the whole stage
            #
            # choose a method to train the interval prediction module
            # split cp
            if args.inv_method == 'split_cp':
                y_pred, _, _, z = model(x)
                q_hat = calibrate_qhat_splitCP(model, train_batch, device, alpha=args.alpha)
                lower, upper = y_pred-q_hat, y_pred+q_hat
                cp_loss = coverage_loss(y, y_pred, lower, upper, args.lamb)
                interval = (abs(upper-lower)/2.5652) ** 2

                nll_loss, nll_mse_component, nll_var_component = compute_point_loss_components(args, y_pred, interval, y, w)
                total_loss = nll_loss
                if args.inv_method == 'split_cp':
                    total_loss = total_loss + args.weight * cp_loss

                opt_extractor.zero_grad()
                opt_regressor.zero_grad()
                total_loss.backward()
                opt_extractor.step()
                opt_regressor.step()
            # cqr : pinball loss based
            elif args.inv_method == 'cqr_pinball':
                y_pred, lower, upper, z = model(x)
                q_hat = calibrate_qhat_from_batch(model, train_batch, device, alpha=args.alpha)

                loss_lower_quantile, loss_upper_quantile = cqr_pinball(y, upper, lower, lamb=args.lamb)
                cp_loss = coverage_loss(y, y_pred, lower, upper, args.lamb)
                interval = ((upper - lower) / 2.5632) ** 2
                interval_loss = interval_minimization(upper, lower)

                total_interval_loss = (loss_lower_quantile + loss_upper_quantile + interval_loss + args.weight * cp_loss)

                nll_loss, nll_mse_component, nll_var_component = compute_point_loss_components(
                    args, y_pred, interval, y, w)

                cp_params = list(model.interval_lower.parameters()) + list(model.interval_upper.parameters())
                main_params = list(model.model_extractor.parameters()) + list(model.pred_head.parameters())

                grads_cp = torch.autograd.grad(
                    total_interval_loss,
                    cp_params,
                    retain_graph=True,
                    allow_unused=False,)
                grads_main = torch.autograd.grad(
                    nll_loss,
                    main_params,
                    retain_graph=False,
                    allow_unused=False,)

                opt_cp_lower.zero_grad()
                opt_cp_upper.zero_grad()
                opt_extractor.zero_grad()
                opt_regressor.zero_grad()

                for p, g in zip(cp_params, grads_cp):
                    p.grad = g

                for p, g in zip(main_params, grads_main):
                    p.grad = g

                opt_cp_lower.step()
                opt_cp_upper.step()
                opt_extractor.step()
                opt_regressor.step()


            elif args.inv_method == 'cqr_coverage':
                #
                y_pred, lower, upper, z = model(x)
                q_hat = calibrate_qhat_from_batch(model, cal_batch, device, alpha=args.alpha)
                cp_loss = coverage_loss(y, y_pred, lower, upper, args.lamb)
                interval = ((upper - lower) / 2.5632) ** 2
                interval_loss = interval_minimization(upper, lower)

                total_interval_loss = args.weight * cp_loss + interval_loss

                nll_loss, nll_mse_component, nll_var_component = compute_point_loss_components(
                    args, y_pred, interval, y, w)

                cp_params = list(model.interval_lower.parameters()) + list(model.interval_upper.parameters())
                main_params = list(model.model_extractor.parameters()) + list(model.pred_head.parameters())

                grads_cp = torch.autograd.grad(
                    total_interval_loss,
                    cp_params,
                    retain_graph=True,
                    allow_unused=False,)
                grads_main = torch.autograd.grad(
                    nll_loss,
                    main_params,
                    retain_graph=False,
                    allow_unused=False,)

                opt_cp_lower.zero_grad()
                opt_cp_upper.zero_grad()
                opt_extractor.zero_grad()
                opt_regressor.zero_grad()

                for p, g in zip(cp_params, grads_cp):
                    p.grad = g

                for p, g in zip(main_params, grads_main):
                    p.grad = g

                opt_cp_lower.step()
                opt_cp_upper.step()
                opt_extractor.step()
                opt_regressor.step()

            #################
            else:
                NotImplementedError

        
            
            
        #
        interval_list.append(interval.detach())
        label_list.append(y.detach())
        pred_list.append(y_pred.detach())
        z_list.append(z.detach())
        if lower is not None and upper is not None:
            lower_list.append(lower.detach())
            upper_list.append(upper.detach())
        nll_loss_history.append(nll_loss.item())
        nll_mse_history.append(nll_mse_component.item())
        nll_var_history.append(nll_var_component.item())
    #
    vars, labels, preds, z_  = torch.cat(interval_list, 0), torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
    epoch_nll_loss = float(np.mean(nll_loss_history)) if nll_loss_history else 0.0
    epoch_nll_mse = float(np.mean(nll_mse_history)) if nll_mse_history else 0.0
    epoch_nll_var = float(np.mean(nll_var_history)) if nll_var_history else 0.0
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
    if lower_list and upper_list:
        lowers, uppers = torch.cat(lower_list, 0), torch.cat(upper_list, 0)
        coverage_maj, coverage_med, coverage_low, coverage_total = compute_interval_coverage(
            lowers, uppers, labels, maj, med, low, device
        )
    else:
        coverage_maj = coverage_med = coverage_low = coverage_total = float('nan')
    #
    pred_results = [
        str(uncer_maj),
        str(uncer_med),
        str(uncer_low),
        str(uncer_total),
        str(epoch_nll_loss),
        str(epoch_nll_mse),
        str(epoch_nll_var),
    ]
    coverage_results = [
        str(coverage_maj),
        str(coverage_med),
        str(coverage_low),
        str(coverage_total),
    ]
    #
    vars_results_from_pred = [str(uncer_pred_maj), str(uncer_pred_med), str(uncer_pred_low), str(uncer_pred_total)]
    #
    

    return model, pred_results, coverage_results, vars_results_from_pred


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
    if args.variance_mse_threshold is not None and args.variance_mse_threshold < 0:
        raise ValueError('--variance_mse_threshold must be non-negative.')
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
    start_epoch = maybe_resume_from_warmup_checkpoint(args, model, opts)
    output_file = 'beta_' + str(args.beta) + '_with_variance_threshold' + str(args.inv_method) + '.txt'
    #output_file = 'nll_output_vs_pred' + '_beta_' + str(args.beta) + '.txt'
    #
    for e in tqdm(range(start_epoch, args.epoch)):
        model, pred_results, coverage_results, vars_results_from_pred = train_one_epoch(args, model, train_loader, val_loader, opts, e)
        save_warmup_checkpoint(args, model, opts, e)
        mae_pred = test(model, test_loader, train_labels, args)
        #
        # record the prediction variance (from predicted labels) and model output variance respectively
        #
        
        with open('nll_' + output_file, "a+") as file:
            file.write(str(e)+" ")
            file.write(" ".join(pred_results) + " " + " ".join(coverage_results) + " " + " ".join(vars_results_from_pred) + " " + str(mae_pred) + '\n')
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
