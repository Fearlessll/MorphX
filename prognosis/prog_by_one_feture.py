import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import optimize

def get_excel_data_TCGA(dataset):
    data = pd.read_csv('HCC_path/TCGA/TCGA.csv')
    ES_data = pd.read_csv('HCC_path/TCGA/clinical.project-tcga-lihc.2025-06-24/clinical.csv')
    censors = []
    survivetimes = []
    stages = []
    ESs = []
    for seg_filepath in dataset:

        #ID = os.path.basename(seg_filepath).split('.')[0][0:12]
        if seg_filepath[0:4] == 'TCGA':
            ID = seg_filepath.split('.')[0][0:23]
        else:
            ID = seg_filepath.split('-')[0]
        #ID = seg_filepath.split('.')[0][0:12]
        #ID = os.path.basename(seg_filepath).split('-')[0]
        try:
            ES_data['WSIs'] = ES_data['WSIs'].astype(str)
            ES_ID = seg_filepath.split('.')[0][0:12]
            ES_pd_index = ES_data[ES_data['WSIs'].isin([ES_ID])].index.values[0]
            ES = ES_data['grade'][ES_pd_index]
        except Exception as e:
            ES = 5
            print(e)

        ESs.append(ES)

        data['WSIs'] = data['WSIs'].astype(str)
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        if data['vital_status'][pd_index] == 1:
            T = data['days_to_last_follow_up'][pd_index] / 30
        else:
            T = data['days_to_death'][pd_index] / 30
        O = (~data['vital_status'][pd_index].astype(bool)).astype(int)
        censors.append(O)
        survivetimes.append(T)
        stages.append(float(data['stage'][pd_index]))
    return censors, survivetimes, stages, ESs

def get_excel_data_CohortLIHC(dataset):
    data = pd.read_csv('HCC_path/CohortLIHC/CohortLIHC_data.csv')
    BCLC_data = pd.read_csv('HCC_path/CohortLIHC/BCLC.csv')
    censors = []
    survivetimes = []
    stages = []
    BCLCs = []
    for seg_filepath in dataset:
        #ID = os.path.basename(seg_filepath).split('-')[0]
        ID = seg_filepath.split('-')[0]
        data['WSIs'] = data['WSIs'].astype(str)
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]

        BCLC_data['WSIs'] = BCLC_data['WSIs'].astype(str)
        try:
            BCLC_pd_index = BCLC_data[BCLC_data['WSIs'].isin([ID])].index.values[0]
            BCLC = BCLC_data['BCLC'][BCLC_pd_index]
        except Exception as e:
            print(e)
            BCLC = 0

        # if data['vital_status'][pd_index] == 1:
        #     T = data['days_to_last_follow_up'][pd_index] / 30
        # else:
        #     T = data['days_to_death'][pd_index] / 30
        T = data['OS'][pd_index] / 30
        O = (data['OS_status'][pd_index].astype(bool)).astype(int)
        censors.append(O)
        survivetimes.append(T)
        stages.append(float(data['stage'][pd_index]))
        BCLCs.append(BCLC)
    return censors, survivetimes, stages, BCLCs




def find_optimal_cutoff(df, time_col='survival_time', event_col='event', score_col='risk_score'):

    def _logrank_pvalue(cutoff):
        groups = df[score_col] > cutoff
        if groups.sum() == 0 or (~groups).sum() == 0:
            return 1.0
        results = logrank_test(
            df[time_col][groups],
            df[time_col][~groups],
            df[event_col][groups],
            df[event_col][~groups]
        )
        return results.p_value

    search_space = np.quantile(df[score_col], np.linspace(0.05, 0.95, 50))
    p_values = np.array([_logrank_pvalue(c) for c in search_space])

    optimal_idx = np.nanargmin(p_values)

    return search_space[optimal_idx]



def km_curve_two_groups(hazardsdata, event_observed, survival_times , data_set_name, time_points=None):

    df = pd.DataFrame({
        'risk_score': hazardsdata,
        'survival_time': survival_times,
        'event': event_observed,
    })

    optimal_cutoff = find_optimal_cutoff(df)
    # optimal_cutoff = np.percentile(hazardsdata, 50)

    print(optimal_cutoff)
    df['risk_group'] = np.where(df['risk_score'] > optimal_cutoff, 'High', 'Low')
    df['group_numeric'] = df['risk_group'].map({'Low': 0, 'High': 1})
    cox = CoxPHFitter()
    df_cox = df[['survival_time', 'event', 'group_numeric']]
    cox.fit(df_cox, duration_col='survival_time', event_col='event')
    hr = cox.summary.loc['group_numeric', 'exp(coef)']
    ci_lower = cox.summary.loc['group_numeric', 'exp(coef) lower 95%']
    ci_upper = cox.summary.loc['group_numeric', 'exp(coef) upper 95%']
    cox_p = cox.summary.loc['group_numeric', 'p']

    plt.figure(figsize=(5, 5))
    ax1 = plt.gca()

    kmf = KaplanMeierFitter()
    colors = ['#4E79A7', '#E15759']
    high_risk = df[df['risk_group'] == 'High']
    kmf.fit(high_risk['survival_time'], event_observed=high_risk['event'], label=f'High DR SMO (n={len(high_risk)})')
    kmf.plot_survival_function(
        ax=ax1, ci_show=False, color=colors[0],
        linewidth=2.5, marker='|', markersize=10
    )
    low_risk = df[df['risk_group'] == 'Low']
    kmf.fit(low_risk['survival_time'], event_observed=low_risk['event'], label=f'Low DR SMO (n={len(low_risk)})')
    kmf.plot_survival_function(
        ax=ax1, ci_show=False, color=colors[1],
        linewidth=2.5, marker='|', markersize=10
    )



    # for i, (name, group) in enumerate(df.groupby('risk_group', sort=False)):
    #     kmf.fit(group['survival_time'], group['event'],
    #             label=f'{name} MCC (n={len(group)})')
    #     kmf.plot_survival_function(
    #         ax=ax1, ci_show=False, color=colors[i],
    #         linewidth=2.5, marker='|', markersize=10
    #     )

    try:
        results = logrank_test(
            df['survival_time'][df['risk_group'] == 'High'],
            df['survival_time'][df['risk_group'] == 'Low'],
            df['event'][df['risk_group'] == 'High'],
            df['event'][df['risk_group'] == 'Low'])
        p_value = results.p_value
    except:
        p_value = np.nan

    # ax1.text(0.4, 0.65,
    #          f'Logrank p = {p_value:.3g}\nHR = {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})',
    #          transform=ax1.transAxes, fontsize=16,
    #          bbox=dict(facecolor='white', alpha=0.8))
    # ax1.text(0.05, 0.1,
    #          f'Logrank p = {p_value:.3g}\nHR = {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})',
    #          transform=ax1.transAxes, fontsize=16,
    #          bbox=dict(facecolor='white', alpha=0.8))
    ax1.text(0.1, 0.55,
             f'Logrank p = {p_value:.3g}\nHR = {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})\nCutoff: {optimal_cutoff:.3g}',
             transform=ax1.transAxes, fontsize=16,
             bbox=dict(facecolor='white', alpha=0.8))
    ax1.set_xlabel('Time (Months)', fontsize=17)
    ax1.set_ylabel('Survival Probability', fontsize=17)
    #ax1.set_title('Kaplan-Meier Survival Curve', fontsize=18, pad=20)
    ax1.tick_params(axis='both', labelsize=14)

    if time_points is None:
        time_points = ax1.get_xticks()[1:-1]
        time_points = time_points[time_points >= 0]
        time_points = np.unique(time_points.astype(int))

    if len(time_points) < 3:
        max_time = int(np.ceil(df['survival_time'].max()))
        time_points = np.linspace(0, max_time, 5).astype(int)
        time_points[-1] = min(time_points[-1], max_time)

    ax1.set_xticks(time_points)

    ax1.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    #ax1.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.5, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'HCC_MCC_km_{data_set_name}.png', dpi=500)
    plt.show()

    return df


from tqdm import tqdm


if __name__ == '__main__':

    data_set_name = 'TCGA'

    feature_maps_dir = f'HCC_path/{data_set_name}/processed_data/feature_maps/concat_feature_maps/168d/initial/final_feature_maps/256'

    data_set_json_path = f'HCC_path/{data_set_name}/all_data.json'

    with open(data_set_json_path, 'r', encoding='utf-8-sig') as f:
        data_set = json.load(f)

    X = []
    for data_name in tqdm(data_set):
        f_maps = np.load(os.path.join(feature_maps_dir, data_name))

        tissue_channels = f_maps[:, :, :8]
        tissue_channels[:, :, [0, 1]] = tissue_channels[:, :, [1, 0]]
        tissue_type_mask = np.argmax(tissue_channels, axis=-1)

        gmcm_MCC = f_maps[:,:,8+19]

        gmcm_MCC = gmcm_MCC[tissue_type_mask == 6]
        gmcm_MCC_maker = np.mean(gmcm_MCC)

        X.append(gmcm_MCC_maker)

    X = np.array(X)

    if data_set_name == 'TCGA':
        test_censors, test_sruvivetimes,stages, ESs = get_excel_data_TCGA(data_set)
    elif data_set_name == 'CohortLIHC':
        test_censors, test_sruvivetimes,ESs, stages = get_excel_data_CohortLIHC(data_set)

    km_curve_two_groups(X, test_censors, test_sruvivetimes, data_set_name)
