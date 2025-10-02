# Env
import statistics

import numpy as np

from data_loaders import *
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from Networks.resnet import resnet10,resnet18, resnext50_32x4d

from Networks.fusion_net import FusionNet

from train_mfm_msfm import test, get_excel_data_TCGA,  get_excel_data_CohortLIHC
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

from utils import cox_loss, modified_cox_loss, cindex_lifeline


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter


def km_curve(hazardsdata, event_observed, survival_times, km_save_path):
    risk_scores = hazardsdata
    survival_time = survival_times
    event = event_observed

    median_risk_score = np.percentile(risk_scores, 50)
    risk_group = ['High' if score >= median_risk_score else 'Low' for score in risk_scores]

    df = pd.DataFrame({
        'risk_score': risk_scores,
        'survival_time': survival_time,
        'event': event,
        'risk_group': risk_group
    })

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    low_risk = df[df['risk_group'] == 'Low']
    kmf.fit(low_risk['survival_time'], event_observed=low_risk['event'], label='Low Risk')
    kmf.plot_survival_function(ci_show=False, color='blue', linewidth=4, marker='|', markersize=16)

    high_risk = df[df['risk_group'] == 'High']
    kmf.fit(high_risk['survival_time'], event_observed=high_risk['event'], label='High Risk')
    kmf.plot_survival_function(ci_show=False, color='red', linewidth=4, marker='|', markersize=16)

    results = logrank_test(low_risk['survival_time'], high_risk['survival_time'], event_observed_A=low_risk['event'],
                           event_observed_B=high_risk['event'])
    p_value = results.p_value

    df['group_numeric'] = df['risk_group'].map({'Low': 0, 'High': 1})

    cox = CoxPHFitter()
    # df_cox = df[['survival_time', 'event', 'risk_score']]
    df_cox = df[['survival_time', 'event', 'group_numeric']]
    cox.fit(df_cox, duration_col='survival_time', event_col='event')
    # cox.print_summary()
    cox_summary = cox.summary
    hr = cox_summary.loc['group_numeric', 'exp(coef)']  # 提取HR
    ci_lower = cox_summary.loc['group_numeric', 'exp(coef) lower 95%']  # 置信区间下限
    ci_upper = cox_summary.loc['group_numeric', 'exp(coef) upper 95%']  # 置信区间上限
    cox_p = cox_summary.loc['group_numeric', 'p']

    low_risk_count = len(low_risk)
    high_risk_count = len(high_risk)

    plt.xlabel('Time (Months)', fontsize=30)
    plt.ylabel('Survival Probability', fontsize=30)
    formatted_p_value = f"{p_value:.2g}"

    textstr = f'HR = {hr:.2f}\n95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]\np = {formatted_p_value}'
    plt.text(0.02, 0.02, textstr, transform=plt.gca().transAxes, fontsize=28, verticalalignment='bottom',bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    plt.legend(labels=[f'Low Risk (n={low_risk_count})', f'High Risk (n={high_risk_count})'], fontsize=28, loc='upper right')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # plt.show()
    plt.savefig(km_save_path)

    cox = CoxPHFitter()
    df_cox = df[['survival_time', 'event', 'risk_score']]
    cox.fit(df_cox, duration_col='survival_time', event_col='event')
    # cox.print_summary()
    cox_summary = cox.summary
    hr = cox_summary.loc['risk_score', 'exp(coef)']  # 提取HR
    ci_lower = cox_summary.loc['risk_score', 'exp(coef) lower 95%']  # 置信区间下限
    ci_upper = cox_summary.loc['risk_score', 'exp(coef) upper 95%']  # 置信区间上限
    cox_p = cox_summary.loc['risk_score', 'p']
    textstr = f'HR = {hr:.2f}\n95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]\np = {cox_p}'
    print(textstr)
    return df
#

if __name__ == '__main__':

    model_name = 'macro'
    data_set_name='TCGA'
    channel_name = 168
    patch_num = 16
    best_macro_th = [5]
    all_data = []
    all_censors = []
    all_sruvivetimes = []
    all_hazard_pred = []
    for fold, th in [(0,5)]:

        base_dir = f'HCC_path/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_name}d/initial/final_feature_maps/256'

        data_set_json_path = f'HCC_path/{data_set_name}/split_data_fold_{fold}.json'

        if model_name == 'fusion':
            patch_feat_path = f'HCC_path/{data_set_name}/topk_tiles_feats/all_wsi_feats_{patch_num}_key_patchs_ori_fold{fold}_{best_macro_th[fold]}th.csv'
            best_model_path = f'train_log_TCGA_256_16_msfm_8+40+120/{fold}/{th}.pkl'

        else:
            best_model_path = f'train_log_TCGA_256_mfm_8+40+120_6.6e-4/{fold}/{th}.pkl'

        with open(data_set_json_path, 'r', encoding='utf-8-sig') as f:
            data_set = json.load(f)
        data_set = data_set["test_data"]
        all_data.extend(data_set)
        test_data = []
        for wsi_name in data_set:
            test_data.append(os.path.join(base_dir, wsi_name))

        transform = A.Compose(
            [
                A.Resize(256, 256),
                ToTensorV2(),
            ]
        )

        if data_set_name == 'TCGA':
            test_censors, test_sruvivetimes = get_excel_data_TCGA([os.path.basename(path) for path in test_data])
            all_censors.extend(test_censors)
            all_sruvivetimes.extend(test_sruvivetimes)
        if model_name == 'fusion':
            model = FusionNet(macro_first_covd_param=[3, 2, 1], macro_input_channel_num=channel_name,
                              patch_first_covd_param=[7, 2, 3], patch_input_channel_num=48,
                              output_use_sigmoid=True)
            test_data_loader = MyFusionDataset(test_data, test_censors, test_sruvivetimes, patch_feat_path, transform=transform)

        else:
            model = resnet10(first_covd_param=[3,2,1], input_channel_num=channel_name,
                            output_use_sigmoid=True)
            test_data_loader = MyDataset(test_data, test_censors, test_sruvivetimes, transform=transform)

        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model, device_ids=[0])
        model = model.to(device)

        test_label_to_count = {}
        for label in test_censors:
            if label not in test_label_to_count:
                test_label_to_count[label] = 0
            test_label_to_count[label] += 1

        test_loader = torch.utils.data.DataLoader(dataset=test_data_loader, batch_size=8,
                                                  shuffle=False, drop_last=False, num_workers=4)

        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, model_name, test_loader, modified_cox_loss, device)
        all_hazard_pred.extend(list(pred_test[0]))

    km_save_path = f'{data_set_name}/result/{model_name}_{channel_name}_{patch_num}.png'

    df_group = km_curve(all_hazard_pred, all_censors, all_sruvivetimes, km_save_path)


