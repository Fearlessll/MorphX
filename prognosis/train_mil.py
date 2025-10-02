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
from Networks.CLAM import CLAM_SB, CLAM_MB
from Networks.transmil import TransMIL
from data_loaders_mil import MyDatasetMIL
from utils import cox_loss, modified_cox_loss, cindex_lifeline, cox_log_rank, accuracy_cox, count_parameters
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import pickle


class MILTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.set_seeds(2024)

    def set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_model(self):
        if self.args.model_name.lower() == 'transmil':
            return TransMIL(ndim=1024, n_classes=1)
        elif self.args.model_name.lower() == 'clam':
            return CLAM_MB(ndim=1024, n_classes=1)
        else:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

    def get_loss_function(self):
        if self.args.loss_function == 'modified_cox_loss':
            return modified_cox_loss
        elif self.args.loss_function == 'cox_loss':
            return cox_loss
        else:
            raise ValueError(f"Unknown loss function: {self.args.loss_function}")

    def get_data_loaders(self, train_data, test_data, train_censors, test_censors, train_survtimes, test_survtimes):
        transform = A.Compose([
            A.Resize(self.args.input_size, self.args.input_size),
            ToTensorV2(),
        ])

        train_dataset = MyDatasetMIL(train_data, train_censors, train_survtimes, transform=transform)
        test_dataset = MyDatasetMIL(test_data, test_censors, test_survtimes, transform=transform)

        # Create weighted samplers
        train_sampler = self._create_sampler(train_censors, train_dataset)
        test_sampler = self._create_sampler(test_censors, test_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        return train_loader, test_loader

    def _create_sampler(self, censors, dataset):
        label_to_count = {}
        for label in censors:
            if label not in label_to_count:
                label_to_count[label] = 0
            label_to_count[label] += 1

        weight_for_0 = len(dataset) / float(label_to_count[0])
        weight_for_1 = len(dataset) / float(label_to_count[1])
        class_weights = [weight_for_0 if label == 0 else weight_for_1 for label in censors]

        return WeightedRandomSampler(class_weights, len(class_weights), replacement=False)

    def train_epoch(self, model, train_loader, optimizer, loss_function):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch = 0

        if self.args.model_name.lower() == 'clam':
            return self._train_epoch_clam(model, train_loader, optimizer, loss_function)
        else:
            return self._train_epoch_standard(model, train_loader, optimizer, loss_function)

    def _train_epoch_standard(self, model, train_loader, optimizer, loss_function):
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch = 0

        for batch_idx, (x_path, survtime, censor) in enumerate(train_loader):
            censor = censor.to(self.device)
            macro_path = x_path.to(self.device).float()

            with autocast(enabled=False):
                _, pred = model(macro_path)
                loss = loss_function(survtime, censor, pred, self.device)

            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        loss_epoch /= len(train_loader.dataset)
        cindex_epoch = cindex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)

        return loss_epoch, cindex_epoch, pvalue_epoch, surv_acc_epoch

    def _train_epoch_clam(self, model, train_loader, optimizer, loss_function):
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch = 0
        accumulation_steps = 10
        accumulated_preds = []
        accumulated_survtime = []
        accumulated_censor = []

        for batch_idx, (x_path, survtime, censor) in enumerate(train_loader):
            survtime = survtime.to(self.device)
            censor = censor.to(self.device)
            x_path = torch.squeeze(x_path, dim=0)
            macro_path = x_path.to(self.device).float()

            _, pred = model(macro_path)

            accumulated_preds.append(pred)
            accumulated_survtime.append(survtime)
            accumulated_censor.append(censor)

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

            if (batch_idx + 1) % accumulation_steps == 0:
                accumulated_preds = torch.cat(accumulated_preds, dim=0)
                accumulated_survtime = torch.cat(accumulated_survtime, dim=0)
                accumulated_censor = torch.cat(accumulated_censor, dim=0)

                loss_cox = cox_loss(accumulated_survtime, accumulated_censor, accumulated_preds, self.device)

                optimizer.zero_grad()
                loss_cox.backward()
                optimizer.step()

                accumulated_preds = []
                accumulated_survtime = []
                accumulated_censor = []
                loss_epoch += loss_cox.data.item()

        if len(accumulated_preds) > 0:
            accumulated_preds = torch.cat(accumulated_preds, dim=0)
            accumulated_survtime = torch.cat(accumulated_survtime, dim=0)
            accumulated_censor = torch.cat(accumulated_censor, dim=0)

            loss_cox = cox_loss(accumulated_survtime, accumulated_censor, accumulated_preds, self.device)

            optimizer.zero_grad()
            loss_cox.backward()
            optimizer.step()
            loss_epoch += loss_cox.data.item()

        loss_epoch /= len(train_loader.dataset)
        cindex_epoch = cindex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)

        return loss_epoch, cindex_epoch, pvalue_epoch, surv_acc_epoch

    def test(self, model, test_loader, loss_function):
        model.eval()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_test = 0

        for batch_idx, (x_path, survtime, censor) in enumerate(test_loader):
            censor = censor.to(self.device)

            if self.args.model_name.lower() == 'clam':
                x_path = torch.squeeze(x_path, dim=0)

            macro_path = x_path.to(self.device).float()

            with autocast(enabled=False):
                _, pred = model(macro_path)

            if loss_function is not None:
                loss = loss_function(survtime, censor, pred, self.device)
                loss_test += loss.data.item()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        loss_test /= len(test_loader.dataset)
        cindex_test = cindex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
        pred_test = [risk_pred_all, survtime_all, censor_all, None, None]

        return loss_test, cindex_test, pvalue_test, surv_acc_test, None, pred_test

    def train(self, train_data, test_data, k_th_fold):
        print(f"Training fold {k_th_fold} on device: {self.device}")

        # Get survival data
        if self.args.data_set_name == 'TCGA':
            train_censors, train_survtimes = self.get_excel_data_TCGA(train_data)
            test_censors, test_survtimes = self.get_excel_data_TCGA(test_data)
        elif self.args.data_set_name == 'CohortLIHC':
            train_censors, train_survtimes = self.get_excel_data_CohortLIHC(train_data)
            test_censors, test_survtimes = self.get_excel_data_CohortLIHC(test_data)


        # Prepare data paths
        train_data = [os.path.join(self.args.final_save_dir, wsi_name.strip('.npy') + '.pt') for wsi_name in train_data]
        test_data = [os.path.join(self.args.final_save_dir, wsi_name.strip('.npy') + '.pt') for wsi_name in test_data]

        # Initialize model and data loaders
        model = self.get_model().to(self.device)
        loss_function = self.get_loss_function()
        train_loader, test_loader = self.get_data_loaders(train_data, test_data, train_censors, test_censors,
                                                          train_survtimes, test_survtimes)

        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=4e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, threshold=0.01, patience=1)
        print(f"Number of Trainable Parameters: {count_parameters(model)}")

        # Initialize metric logger
        metric_logger = {
            'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
            'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
            'val': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []}
        }

        cindex_test_max = 0

        for epoch in tqdm(range(self.args.epochs)):
            # Training
            train_loss, train_cindex, train_pvalue, train_surv_acc = self.train_epoch(
                model, train_loader, optimizer, loss_function)

            # Testing
            test_loss, test_cindex, test_pvalue, test_surv_acc, _, test_pred = self.test(
                model, test_loader, loss_function)

            # Validation (optional)
            val_loss, val_cindex, val_pvalue, val_surv_acc, _, val_pred = self.test(
                model, test_loader, loss_function)  # Using test as val for simplicity

            # Update scheduler
            scheduler.step(train_loss)
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate = {lr:.7f}')
            print('-' * 90)

            # Log metrics
            metric_logger['train']['loss'].append(train_loss)
            metric_logger['train']['cindex'].append(train_cindex)
            metric_logger['train']['pvalue'].append(train_pvalue)
            metric_logger['train']['surv_acc'].append(train_surv_acc)

            metric_logger['test']['loss'].append(test_loss)
            metric_logger['test']['cindex'].append(test_cindex)
            metric_logger['test']['pvalue'].append(test_pvalue)
            metric_logger['test']['surv_acc'].append(test_surv_acc)

            metric_logger['val']['loss'].append(val_loss)
            metric_logger['val']['cindex'].append(val_cindex)
            metric_logger['val']['pvalue'].append(val_pvalue)
            metric_logger['val']['surv_acc'].append(val_surv_acc)

            # Print progress
            self._print_epoch_progress('Train', train_loss, train_surv_acc, train_cindex, train_pvalue)
            self._print_epoch_progress('Test', test_loss, test_surv_acc, test_cindex, test_pvalue)
            self._print_epoch_progress('Val', val_loss, val_surv_acc, val_cindex, val_pvalue)

            # Save checkpoint
            save_path = os.path.join(
                self.args.train_log_dir,
                f'train_log_{self.args.data_set_name}_{self.args.input_size}_{self.args.model_name}_{self.args.input_data_type}/{k_th_fold}th'
            )
            os.makedirs(save_path, exist_ok=True)

            if test_cindex > cindex_test_max:
                cindex_test_max = test_cindex

            torch.save({
                'split': k_th_fold,
                'epoch': epoch,
                'data': [train_data, test_data],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metric_logger},
                save_path + f'/{epoch}.pkl'
            )

            pickle.dump(test_pred, open(save_path + f'/pred_test_{k_th_fold}.pkl', 'wb'))

        return model, optimizer, metric_logger

    def _print_epoch_progress(self, phase, loss, surv_acc, cindex, pvalue):
        print(f'[{phase}]\t\tLoss: {loss:.4f}, surv_acc: {surv_acc:.4f}, C-Index: {cindex:.4f}, p-value: {pvalue}')

    # Data loading methods (get_excel_data_*) remain the same as in original code
    def get_excel_data_TCGA(self, dataset):
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

    def get_excel_data_CohortLIHC(self, dataset):
        data = pd.read_csv(
            'HCC_path/CohortLIHC/CohortLIHC_data.csv')
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


def train_k_folds(args):

    args.split_data_dir = f'{args.base_dir}/{args.data_set_name}'
    trainer = MILTrainer(args)

    for fold in range(args.k_fold):
        with open(os.path.join(args.split_data_dir, f'split_data_fold_{fold}.json'), 'r', encoding='utf-8-sig') as f:
            data_set = json.load(f)

        train_data = data_set["train_data"]
        test_data = data_set["test_data"]

        print(f'fold_{fold}:')
        print(f'Train samples: {len(train_data)}')
        print(f'Test samples: {len(test_data)}')

        trainer.train(train_data, test_data, fold)

    calculate_average_cindex(args)


def calculate_average_cindex(args):
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
        length = len(cindex_list)
        flag = False

        for j in range(length):
            max_cindex_value = max(cindex_list)
            index = cindex_list.index(max_cindex_value) + start
            if pvalue_list[index] < 0.95:
                max_cindex_list.append(max_cindex_value)
                max_cindex_value_index = cindex_list.index(max_cindex_value) + start
                max_cindex_index_list.append((round(max_cindex_value, 4), max_cindex_value_index))
                flag = True
                break
            else:
                del cindex_list[index]
                del pvalue_list[index]

        if not flag:
            max_cindex_index_list.append((0, 0))

        epoch_last_pkl = None
        torch.cuda.empty_cache()

    print('Max C-index values:')
    print(max_cindex_list)
    print('Max C-index indices:')
    print(max_cindex_index_list)
    print(f'Average C-index: {round(sum(max_cindex_list) / len(max_cindex_list), 4)}')
    print(f'Standard deviation: {round(statistics.stdev(max_cindex_list), 4)}')


def parse_args():
    parser = argparse.ArgumentParser(description="WSI Training")

    parser.add_argument("--input_data_type", default='8', help="Input data type")
    parser.add_argument("--input_size", default=256, type=int, help="Input size")
    parser.add_argument("--nor_method", default='initial', help="Normalization method")

    parser.add_argument("--model_name", default="transmil", help="Model type (transmil or clam)")
    parser.add_argument("--loss_function", default="cox_loss", help="Loss function (cox_loss or modified_cox_loss)")

    parser.add_argument("--base_dir", default='HCC_path', help="base_dir directory")
    parser.add_argument("--train_log_dir", default='HCC_path', help="Training log directory")
    parser.add_argument("--final_save_dir",
                        default='HCC_path/MIL_256_level1/TCGA/patch_feat_uni/pt_files',
                        help="tile features directory (uni or resnet)")

    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--lr", default=6.6e-3, type=float, help="Learning rate")
    parser.add_argument("--patience", default=1, type=float, help="Patience for scheduler")

    parser.add_argument("--data_set_name", default='TCGA', help="Dataset name")
    parser.add_argument("--k_fold", default=10, type=int, help="Number of folds")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_args()
    train_k_folds(args)