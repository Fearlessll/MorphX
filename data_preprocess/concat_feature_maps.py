import glob
import os
import numpy as np
import json


def concat_feature_maps(wsi_names, tissue_type_maps_dir, feature_maps_dir_list, save_features_concat_dir, use_tissue):

    os.makedirs(save_features_concat_dir, exist_ok=True)

    finished_maps = glob.glob(save_features_concat_dir+'/*')
    for wsi_name in wsi_names:
        if any(wsi_name in s for s in finished_maps) == True:
            print(wsi_name + ' has finished!')
            continue
        try:
            tissue_type_maps = np.load(os.path.join(tissue_type_maps_dir, wsi_name))
        except Exception as e:
            print(e)
            continue
        tissue_type_maps_row, tissue_type_maps_col = tissue_type_maps.shape[0], tissue_type_maps.shape[1]
        if use_tissue == True:
            tissue_type_maps_list = [tissue_type_maps[:, :, i] for i in range(tissue_type_maps.shape[-1])]
        else:
            tissue_type_maps_list = []

        for feature_maps_dir in feature_maps_dir_list:
            feature_imgs = np.load(os.path.join(feature_maps_dir, wsi_name))
            feature_imgs_row, feature_imgs_col = feature_imgs.shape[0], feature_imgs.shape[1]
            if tissue_type_maps_row != feature_imgs_row or tissue_type_maps_col != feature_imgs_col:
                feature_imgs = np.resize(feature_imgs, (tissue_type_maps_row, tissue_type_maps_col, feature_imgs.shape[-1]))
            feature_imgs_list = [feature_imgs[:, :, i] for i in range(feature_imgs.shape[-1])]
            tissue_type_maps_list.extend(feature_imgs_list)

        concated_features = np.stack(tissue_type_maps_list, axis=2)
        np.save(os.path.join(save_features_concat_dir, wsi_name), concated_features)


if __name__ == '__main__':

    data_set = 'TCGA'
    base_dir = f'HCC_path/{data_set}'

    channel_num = 160
    use_tissue = True
    if channel_num == 48:
        feature_maps_dir_list = [
            f'{base_dir}/processed_data/feature_maps/hand-crafted/texture_feature_maps']
    elif channel_num == 168:
        feature_maps_dir_list = [f'{base_dir}/processed_data/feature_maps/hand-crafted/texture_feature_maps',
                                 f'{base_dir}/processed_data/feature_maps/hand-crafted/nucleus_feature_maps']
    elif channel_num == 160:
        feature_maps_dir_list = [f'{base_dir}/processed_data/feature_maps/hand-crafted/texture_feature_maps',
                                 f'{base_dir}/processed_data/feature_maps/hand-crafted/nucleus_feature_maps']
        use_tissue = False
    elif channel_num == 128:
        feature_maps_dir_list = [
                                 f'{base_dir}/processed_data/feature_maps/hand-crafted/nucleus_feature_maps']

    tissue_type_maps_dir = f'{base_dir}/processed_data/feature_maps/tissue_type/tissue_type_maps'
    tissue_mask_dir = f'{base_dir}/processed_data/histoqc'
    save_features_concat_dir = f'{base_dir}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/initial/concat_feature_maps'
    with open(os.path.join(f'{base_dir}/all_data.json'), 'r', encoding='utf-8-sig') as f:
        wsi_names = json.load(f)

    concat_feature_maps(wsi_names, tissue_type_maps_dir, feature_maps_dir_list, save_features_concat_dir, use_tissue)






