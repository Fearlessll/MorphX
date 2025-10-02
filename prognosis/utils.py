import math
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def nll_loss(hazards, S, Y, c, alpha=0.15, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def cox_loss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).float()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    #median = 0.5
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1

    # hazardsdata_sorted = np.sort(hazardsdata)
    # median = hazardsdata_sorted[len(hazardsdata) // 3 * 2]
    # hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    # hazards_dichotomize[hazardsdata < median] = 1

    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def cox_log_rank_plot_hr(hazardsdata, labels, survtime_all):
    df = pd.DataFrame({
        'duration': survtime_all,
        'event': labels,
        'risk_score': hazardsdata
    })

    median_risk_score = df['risk_score'].median()
    df['group'] = np.where(df['risk_score'] >= median_risk_score, 'High Risk', 'Low Risk')

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 6))

    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        kmf.fit(group_data['duration'], event_observed=group_data['event'], label=group)
        kmf.plot_survival_function()

    plt.title('Kaplan-Meier Survival Curves by Risk Group')
    plt.xlabel('Survival Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.show()

    low_risk = df[df['group'] == 'Low Risk']
    high_risk = df[df['group'] == 'High Risk']

    logrank_result = logrank_test(low_risk['duration'], high_risk['duration'],
                                  event_observed_A=low_risk['event'], event_observed_B=high_risk['event'])

    print(f"Log-rank test p-value: {logrank_result.p_value}")

    df['group_numeric'] = df['group'].map({'Low Risk': 0, 'High Risk': 1})
    cph = CoxPHFitter()
    cph.fit(df[['duration', 'event', 'group_numeric']], duration_col='duration', event_col='event')

    cph.print_summary()

def cox_log_rank_plot(hazardsdata, labels, survtime_all):
    data = {
        'predicted_risk_score': hazardsdata,
        'survival_time': survtime_all,
        'event_observed': labels
    }
    df = pd.DataFrame(data)


    #threshold = np.percentile(df['predicted_risk_score'], 75)
    threshold = df['predicted_risk_score'].median()
    df['risk_group'] = np.where(df['predicted_risk_score'] > threshold, 'high', 'low')

    plt.figure(figsize=(10, 6))

    for risk_group, group_df in df.groupby('risk_group'):
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['survival_time'], event_observed=group_df['event_observed'], label=risk_group)
        if risk_group == 'high':
            color = 'r'
        else:
            color = 'b'
        kmf.plot(ax=plt.gca(), legend=True, color=color)


    plt.title('Kaplan-Meier Curve by Predicted Risk Group')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.show()

    result = logrank_test(df['survival_time'][df['risk_group'] == 'low'],
                          df['survival_time'][df['risk_group'] == 'high'],
                          event_observed_A=df['event_observed'][df['risk_group'] == 'low'],
                          event_observed_B=df['event_observed'][df['risk_group'] == 'high'])

    print(f"Log-rank test p-value: {result.p_value}")

    cph = CoxPHFitter()
    cph.fit(df, duration_col='survival_time', event_col='event_observed', formula='risk_group')
    cph.print_summary()

    # hazard_ratio = np.exp(cph.summary.loc['risk_group', 'coef'])
    # lower_ci = np.exp(cph.summary.loc['risk_group', 'lower'])
    # upper_ci = np.exp(cph.summary.loc['risk_group', 'upper'])
    #
    # print(f"Hazard Ratio (High-risk vs Low-risk): {hazard_ratio}")
    # print(f"95% Confidence Interval: ({lower_ci}, {upper_ci})")


def plot_km_curve(hazardsdata, labels, survtime_all):
    data = {
        'risk_score': hazardsdata,
        'survival_time': survtime_all,
        'event_observed': labels
    }
    df = pd.DataFrame(data)
    df['risk_group'] = np.where(df['risk_score'] > 0.5, 'high_risk', 'low_risk')

    plt.figure(figsize=(10, 6))

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    kmf_high.fit(df[df['risk_group'] == 'high_risk']['survival_time'],
                 event_observed=df[df['risk_group'] == 'high_risk']['event_observed'],
                 label='High Risk')
    kmf_low.fit(df[df['risk_group'] == 'low_risk']['survival_time'],
                event_observed=df[df['risk_group'] == 'low_risk']['event_observed'],
                label='Low Risk')

    kmf_high.plot(ax=plt.gca(), color='r')
    kmf_low.plot(ax=plt.gca(), color='b')

    plt.title('Kaplan-Meier Curve by Risk Group')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.show()


def cindex_lifeline(hazards, labels, survtime_all):
    return (concordance_index(survtime_all, -hazards, labels))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)