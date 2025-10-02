import statistics
import json
import random
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
from Networks.darknet import darknet53
from Networks.vit import VisionTransformer
from Networks.resnet import resnet10, resnext50_32x4d, resnet18
from Networks.resnet_norm import resnet9 as resnet_norm
from Networks.resnet3D import resnet10 as resnet3d
from Networks.CNN import SimpleCNN
from Networks.fusion_net import FusionNet
from data_loaders import MyDataset, MyFusionDataset
from utils import cox_loss, modified_cox_loss, cindex_lifeline, cox_log_rank, accuracy_cox, count_parameters
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import pickle


def get_excel_data_TCGA(dataset):
    data = pd.read_csv('HCC_path/TCGA/TCGA.csv')
    censors = []
    survivetimes = []
    for seg_filepath in dataset:
        if seg_filepath[0:4] == 'TCGA':
            ID = seg_filepath.split('.')[0][0:23]
        else:
            ID = seg_filepath.split('-')[0]
        data['WSIs'] = data['WSIs'].astype(str)
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        if data['vital_status'][pd_index] == 1:
            T = data['days_to_last_follow_up'][pd_index] / 30
        else:
            T = data['days_to_death'][pd_index] / 30
        O = (~data['vital_status'][pd_index].astype(bool)).astype(int)
        censors.append(O)
        survivetimes.append(T)
    return censors, survivetimes


def get_excel_data_CohortLIHC(dataset):
    data = pd.read_csv('HCC_path/CohortLIHC/CohortLIHC_data.csv')
    censors = []
    survivetimes = []
    for seg_filepath in dataset:
        ID = seg_filepath.split('-')[0]
        data['WSIs'] = data['WSIs'].astype(str)
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        T = data['OS'][pd_index] / 30
        O = (data['OS_status'][pd_index].astype(bool)).astype(int)
        censors.append(O)
        survivetimes.append(T)
    return censors, survivetimes


def get_dataset_survival_data(dataset, data_set_name):
    """Get survival data based on dataset name"""
    if data_set_name == 'TCGA':
        return get_excel_data_TCGA(dataset)
    elif data_set_name == 'CohortLIHC':
        return get_excel_data_CohortLIHC(dataset)
    else:
        raise ValueError(f"Unknown dataset name: {data_set_name}")


def initialize_model(args):
    """Initialize model based on model type (macro or fusion)"""
    if args.model_name == 'macro':
        model = resnet10(
            first_covd_param=args.macro_first_covd_param,
            input_channel_num=args.macro_input_channel_num,
            output_use_sigmoid=args.output_use_sigmoid
        )
    elif args.model_name == 'fusion':
        model = FusionNet(
            macro_first_covd_param=args.macro_first_covd_param,
            macro_input_channel_num=args.macro_input_channel_num,
            output_use_sigmoid=args.output_use_sigmoid,
            macro_best_ckpt_path=args.macro_best_ckpt_path
        )
        if args.freeze_macro_part:
            for name, param in model.macro_net.named_parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    return model.to(args.device)


def initialize_data_loaders(train_data, test_data, train_censors, train_sruvivetimes,
                            test_censors, test_sruvivetimes, args):
    """Initialize data loaders based on model type"""
    transform = A.Compose([
        A.Resize(args.input_size, args.input_size),
        ToTensorV2(),
    ])

    # Prepare full paths
    train_data = [os.path.join(args.final_save_dir, wsi_name) for wsi_name in train_data]
    test_data = [os.path.join(args.final_save_dir, wsi_name) for wsi_name in test_data]

    if args.model_name == 'macro':
        train_dataset = MyDataset(train_data, train_censors, train_sruvivetimes, transform=transform)
        test_dataset = MyDataset(test_data, test_censors, test_sruvivetimes, transform=transform)
    elif args.model_name == 'fusion':
        train_dataset = MyFusionDataset(
            train_data, train_censors, train_sruvivetimes,
            args.patchs_feats_file if hasattr(args, 'patchs_feats_file') else args.mircro_dir,
            args.macro_input_channel_num,
            transform=transform
        )
        test_dataset = MyFusionDataset(
            test_data, test_censors, test_sruvivetimes,
            args.patchs_feats_file if hasattr(args, 'patchs_feats_file') else args.mircro_dir,
            args.macro_input_channel_num,
            transform=transform
        )

    # Create weighted samplers for class imbalance
    def create_sampler(dataset, censors):
        label_to_count = {}
        for label in censors:
            label_to_count[label] = label_to_count.get(label, 0) + 1
        weight_for_0 = len(dataset) / float(label_to_count[0])
        weight_for_1 = len(dataset) / float(label_to_count[1])
        class_weights = [weight_for_0 if label == 0 else weight_for_1 for label in censors]
        return WeightedRandomSampler(class_weights, len(class_weights), replacement=False)

    train_sampler, _ = create_sampler(train_dataset, train_censors)
    test_sampler, _ = create_sampler(test_dataset, test_censors)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, loss_function, optimizer, scheduler, args):
    """Train model for one epoch"""
    model.train()
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_epoch = 0

    for batch_idx, (x_path, survtime, censor) in enumerate(train_loader):
        censor = censor.to(args.device)

        if args.model_name == "macro":
            macro_path = x_path.to(args.device).float()
            with autocast(enabled=False):
                _, pred = model(macro_path)
                loss_cox = loss_function(survtime, censor, pred, args.device)
        elif args.model_name == "fusion":
            imgs_path = x_path[0].to(args.device).float()
            macro_path = x_path[1].to(args.device).float()
            with autocast(enabled=False):
                pred = model(imgs_path, macro_path)
                loss_cox = loss_function(survtime, censor, pred, args.device)

        loss = loss_cox
        loss_epoch += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging information
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    scheduler.step(loss_epoch)
    lr = optimizer.param_groups[0]['lr']
    print(f'learning rate = {lr:.7f}')
    print('-----------------------------------------------------------------------------------------')

    loss_epoch /= len(train_loader.dataset)
    cindex_epoch = cindex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)

    return loss_epoch, cindex_epoch, pvalue_epoch, surv_acc_epoch


def test(model, test_loader, loss_function, args):
    """Evaluate model on test set"""
    model.eval()
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0

    with torch.no_grad():
        for batch_idx, (x_path, survtime, censor) in enumerate(test_loader):
            censor = censor.to(args.device)

            if args.model_name == "macro":
                macro_path = x_path.to(args.device).float()
                with autocast(enabled=False):
                    _, pred = model(macro_path)
                    loss_cox = loss_function(survtime, censor, pred, args.device)
            elif args.model_name == "fusion":
                imgs_path = x_path[0].to(args.device).float()
                macro_path = x_path[1].to(args.device).float()
                with autocast(enabled=False):
                    pred = model(imgs_path, macro_path)
                    loss_cox = loss_function(survtime, censor, pred, args.device)

            loss_test += loss_cox.data.item()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    loss_test /= len(test_loader.dataset)
    cindex_test = cindex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)

    pred_test = [risk_pred_all, survtime_all, censor_all, None, None]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, None, pred_test


def train(train_data, test_data, k_th_fold, args):
    """Main training function for one fold"""
    print(args.device)
    cindex_test_max = 0

    seed = int(2024)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get survival data
    train_censors, train_sruvivetimes = get_dataset_survival_data(train_data, args.data_set_name)
    test_censors, test_sruvivetimes = get_dataset_survival_data(test_data, args.data_set_name)

    # Initialize model and data loaders
    model = initialize_model(args)
    train_loader, test_loader = initialize_data_loaders(
        train_data, test_data,
        train_censors, train_sruvivetimes,
        test_censors, test_sruvivetimes,
        args
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, threshold=0.01, patience=1)
    print(f"Number of Trainable Parameters: {count_parameters(model)}")

    # Initialize metric logger
    metric_logger = {
        'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
        'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
        'val': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []}
    }

    # Loss function
    loss_function = modified_cox_loss if args.loss_function == 'modified_cox_loss' else cox_loss

    # Training loop
    transform = A.Compose([A.Resize(args.input_size, args.input_size), ToTensorV2()])

    for epoch in tqdm(range(args.epochs)):
        # Freeze macro part if needed (for fusion model)
        if args.model_name == 'fusion' and args.freeze_macro_part:
            for name, param in model.macro_net.named_parameters():
                param.requires_grad = False

        # Train epoch
        train_loss, train_cindex, train_pvalue, train_surv_acc = train_epoch(
            model, train_loader, loss_function, optimizer, scheduler, args
        )

        # Evaluate on test set
        test_loss, test_cindex, test_pvalue, test_surv_acc, _, pred_test = test(
            model, test_loader, loss_function, args
        )


        # Update metric logger
        metric_logger['train']['loss'].append(train_loss)
        metric_logger['train']['cindex'].append(train_cindex)
        metric_logger['train']['pvalue'].append(train_pvalue)
        metric_logger['train']['surv_acc'].append(train_surv_acc)

        metric_logger['test']['loss'].append(test_loss)
        metric_logger['test']['cindex'].append(test_cindex)
        metric_logger['test']['pvalue'].append(test_pvalue)
        metric_logger['test']['surv_acc'].append(test_surv_acc)

        # Print metrics
        print(
            f"[Train]\t\tLoss: {train_loss:.4f}, surv_acc: {train_surv_acc:.4f}, C-Index: {train_cindex:.4f}, p-value: {train_pvalue}")
        print(
            f"[Test]\t\tLoss: {test_loss:.4f}, surv_acc: {test_surv_acc:.4f}, C-Index: {test_cindex:.4f}, p-value: {test_pvalue}\n")

        # Save checkpoint
        save_path = os.path.join(
            args.train_log_dir,
            f'train_log_{args.data_set_name}_{args.input_size}_{args.model_name}_{args.input_data_type}/'
            f'{k_th_fold}th'
        )
        os.makedirs(save_path, exist_ok=True)

        if test_cindex > cindex_test_max:
            cindex_test_max = test_cindex

        torch.save({
            'split': k_th_fold,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger
        }, save_path + f'/{epoch}.pkl')

        pickle.dump(pred_test, open(save_path + f'/pred_test_{k_th_fold}.pkl', 'wb'))

    return model, optimizer, metric_logger


def calculate_average_cindex(args):
    """Calculate average C-index across all folds"""
    train_log_base_dir = os.path.join(
        args.train_log_dir,
        f'train_log_{args.data_set_name}_{args.input_size}_{args.model_name}_{args.input_data_type}'
    )
    max_cindex_list = []
    max_cindex_index_list = []

    for i in range(args.k_fold):
        epoch_last_pkl = torch.load(os.path.join(train_log_base_dir, f'{i}th/{args.epochs - 1}.pkl'))
        start = 0
        cindex_list = epoch_last_pkl['metrics']['test']['cindex'][start:]
        pvalue_list = epoch_last_pkl['metrics']['test']['pvalue'][start:]

        flag = False
        for j in range(len(cindex_list)):
            max_cindex_value = max(cindex_list)
            index = cindex_list.index(max_cindex_value) + start
            if pvalue_list[index] < 0.95:
                max_cindex_list.append(max_cindex_value)
                max_cindex_index_list.append((round(max_cindex_value, 4), index))
                flag = True
                break
            else:
                del cindex_list[index]
                del pvalue_list[index]

        if not flag:
            max_cindex_index_list.append((0, 0))

        epoch_last_pkl = None
        torch.cuda.empty_cache()

    print('max_cindex')
    print(max_cindex_list)
    print(max_cindex_index_list)
    print(f"Average: {round(sum(max_cindex_list) / len(max_cindex_list), 4)}")
    print(f"Std: {round(statistics.stdev(max_cindex_list), 4)}")


def train_k_folds(args):
    """Train model using k-fold cross validation"""
    data_set_dir_list = [
        "8", "40", "8+40", "8+120", "120", "40+120", "8+40+120"
    ]

    if args.input_data_type not in data_set_dir_list:
        raise ValueError(f"Invalid input_data_type. Must be one of: {data_set_dir_list}")

    feature_num_arr = [int(f_num) for f_num in args.input_data_type.split('+')]
    args.macro_input_channel_num = sum(feature_num_arr)

    args.split_data_dir = f'{args.base_dir}/{args.data_set_name}'

    # Parse best macro models from args
    if args.model_name == 'fusion':
        if not args.best_macro_models:
            raise ValueError("For fusion model, best_macro_models must be provided")

        # Parse string like "0:5,1:6,2:2,3:13,4:13,5:10,6:14,7:12,8:7,9:13"
        best_macro_model = []
        for pair in args.best_macro_models.split(','):
            fold, epoch = pair.split(':')
            best_macro_model.append((int(fold), int(epoch)))
    else:
        best_macro_model = None

    for fold in range(args.k_fold):
        # Setup paths based on model type
        if args.model_name == 'fusion':
            args.patchs_feats_file = f'{args.base_dir}/{args.data_set_name}/topk_tiles_feats/all_wsi_feats_{args.patch_num}_key_patchs_ori_fold{best_macro_model[fold][0]}_{best_macro_model[fold][1]}th.csv'
            args.macro_best_ckpt_path = f'{args.macro_ckpt_base_dir}/{best_macro_model[fold][0]}th/{best_macro_model[fold][1]}.pkl'

        # Setup feature maps directory
        if args.nor_method == 'initial':
            args.final_save_dir = f'{args.base_dir}/{args.data_set_name}/processed_data/feature_maps/concat_feature_maps/{args.macro_input_channel_num}d/initial/final_feature_maps/{args.input_size}'
        else:
            args.final_save_dir = f'{args.base_dir}/{args.data_set_name}/processed_data/feature_maps/concat_feature_maps/{args.macro_input_channel_num}d/{args.nor_method}/fold_{fold}_final_feature_maps'

        # Load split data
        with open(os.path.join(args.split_data_dir, f'split_data_fold_{fold}.json'), 'r', encoding='utf-8-sig') as f:
            data_set = json.load(f)

        train_data = data_set["train_data"]
        test_data = data_set["test_data"]

        print(f'fold_{fold}:')
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        train(train_data, test_data, fold, args)

    calculate_average_cindex(args)


def parse_args():

    parser = argparse.ArgumentParser(description="WSI Survival Analysis")

    # Data parameters
    parser.add_argument("--input_data_type", default='8+40+120',
                        help="Input data type/feature combination")
    parser.add_argument("--input_size", default=256, type=int,
                        help="Input image size")
    parser.add_argument("--nor_method", default='initial',
                        help="Normalization method: initial, maxmin, zscore")
    parser.add_argument("--data_set_name", default='TCGA',
                        help="Dataset name")
    parser.add_argument("--k_fold", default=10, type=int,
                        help="Number of folds for cross-validation")
    parser.add_argument("--base_dir",
                        default='HCC_path',
                        help="Path to base data directory")
    parser.add_argument("--macro_ckpt_base_dir",
                        default='train_log_TCGA_256_macro_8+40+120_6.6e-4',
                        help="Base directory for macro model checkpoints")

    # Model parameters
    parser.add_argument("--model_name", default="macro",
                        choices=["macro", "fusion"],
                        help="Model type")
    parser.add_argument("--freeze_macro_part", default=False, type=bool,
                        help="Freeze macro part in fusion model")
    parser.add_argument("--macro_first_covd_param", nargs='+', type=int,
                        default=[3, 2, 1],
                        help="Macro model first conv params")
    parser.add_argument("--macro_input_channel_num", default=168, type=int,
                        help="Macro model input channels")
    parser.add_argument("--output_use_sigmoid", default=True, type=bool,
                        help="Use sigmoid in output")
    parser.add_argument("--loss_function", default="modified_cox_loss",
                        choices=["modified_cox_loss", "cox_loss"],
                        help="Loss function")
    parser.add_argument("--patch_num", default=64, type=int,
                        help="Number of patches (for fusion model)")
    parser.add_argument("--best_macro_models",
                        default="0:5,1:6,2:2,3:13,4:13,5:10,6:14,7:12,8:7,9:13",
                        help="Best macro models for fusion training as 'fold:epoch' pairs")

    # Path parameters
    parser.add_argument("--macro_best_ckpt_path", default=None,
                        help="Path to best macro model checkpoint (for fusion)")
    parser.add_argument("--patchs_feats_file", default='',
                        help="Path to patch features file (for fusion)")
    parser.add_argument("--mircro_dir", default='',
                        help="Alternative path to micro features (for fusion)")
    parser.add_argument("--train_log_dir",
                        default='data',
                        help="Training log directory")
    parser.add_argument("--final_save_dir", default='',
                        help="Final feature maps directory")

    # Training parameters
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", default=10, type=int,
                        help="Batch size")
    parser.add_argument("--lr", default=6.6e-4, type=float,
                        help="Learning rate")
    parser.add_argument("--patience", default=1, type=float,
                        help="Patience for LR scheduler")

    # Device
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.model_name == 'fusion':
        for patch_num in [args.patch_num]:  # Now using the patch_num from args
            args.patch_num = patch_num
            train_k_folds(args)
    else:
        train_k_folds(args)