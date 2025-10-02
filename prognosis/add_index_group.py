import json
import pandas as pd
import numpy as np

def compute_statistics_per_group(X, group_size, patch_num):
    n_samples, n_features = X.shape
    n_groups = n_samples // group_size
    new_features = np.zeros((n_groups, n_features))
    for i in range(n_groups):
        group_data = X[i * group_size:(i + 1) * group_size-(group_size-patch_num), :]
        if np.sum(np.isnan(group_data)) > 0:
            print(1)
        clear_group_data = group_data[~np.isnan(group_data).any(axis=1)]
        clear_group_data = clear_group_data[~(np.sum(clear_group_data, axis=1) == 0)]

        clear_group_data = clear_group_data[(clear_group_data ==0).sum(axis=1)<50]
        means = np.mean(clear_group_data, axis=0)

        new_features[i, :] = means
    return new_features

for data_set_name in ['TCGA']:
    for fold, th in [(0, 11)]:
        for patch_num in [16]:

            data_set_json_path = f'HCC_path/{data_set_name}/all_data.json'
            with open(data_set_json_path, 'r', encoding='utf-8-sig') as f:
                data_set = json.load(f)

            features = np.load(f'HCC_path/{data_set_name}/topk_tiles_feats/all_wsi_feats_64_key_patchs_fold{fold}_{th}th.npy')
            zero_rows = np.all(features == 0, axis=1)
            num_zero_rows = np.sum(zero_rows)

            features = compute_statistics_per_group(features, 64, patch_num)

            features = features.astype(float)
            features_pd = pd.DataFrame(features, columns=[f'feature_{i + 1}' for i in range(168)])
            features_pd['wsi_id'] = [data_set[i] for i in range(len(data_set))]

            features_pd.to_csv(f'HCC_path/{data_set_name}/topk_tiles_feats/all_wsi_feats_{patch_num}_key_patchs_ori_fold{fold}_{th}th.csv', index=False)
