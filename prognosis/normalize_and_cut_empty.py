
import glob
import os
import json
import cv2 as cv
from tqdm import tqdm
import numpy as np
import albumentations as A
import gc


def cut_empty_and_resize(matrix, mask, wsi_name, cut_empty_dir, save_base_dir, final_size=256):

    # Crop empty regions from feature matrix and resize to target dimensions.

    # Remove non-tissue regions
    high, weight, channel_num = matrix.shape
    matrix[mask == 0] = 0

    # Find bounding box of tissue region
    coor = np.where(mask != 0)
    top = min(coor[0])
    bottom = max(coor[0])
    left = min(coor[1])
    right = max(coor[1])

    # Check for empty middle sections (common in tissue samples)
    is_width_middle_empty = False
    for w in range(left, right):
        if not is_width_middle_empty and np.sum(mask[:, w]) == 0:
            start_w = w
            is_width_middle_empty = True
        if is_width_middle_empty and np.sum(mask[:, w]) > 0:
            end_w = w
            break

    is_high_middle_empty = False
    for h in range(top, bottom):
        if not is_high_middle_empty and np.sum(mask[h, :]) == 0:
            start_h = h
            is_high_middle_empty = True
        if is_high_middle_empty and np.sum(mask[h, :]) > 0:
            end_h = h
            break

    # Crop matrix based on empty regions
    if is_width_middle_empty:
        matrix = np.concatenate([matrix[:, left:start_w], matrix[:, end_w:right]], axis=1)
    else:
        matrix = matrix[:, left:right]

    if is_high_middle_empty:
        matrix = np.concatenate([matrix[top:start_h, :], matrix[end_h:bottom, :]], axis=0)
    else:
        matrix = matrix[top:bottom, :]

    # Pad to square and center tissue
    height, width, _ = matrix.shape
    if height >= width:
        new_size = height + 8
        new_matrix = np.zeros((new_size, new_size, channel_num))
        dealt_width = int((height - width) / 2)
        new_matrix[4:-4, dealt_width + 4:width + (dealt_width + 4)] = matrix
    else:
        new_size = width + 8
        new_matrix = np.zeros((new_size, new_size, channel_num))
        dealt_height = int((width - height) / 2)
        new_matrix[dealt_height + 4:height + (dealt_height + 4), 4:-4] = matrix

    # Save intermediate cropped matrix
    os.makedirs(cut_empty_dir, exist_ok=True)
    np.save(f'{cut_empty_dir}/{wsi_name}.npy', new_matrix)

    # Resize to final dimensions
    transform = A.Compose([A.Resize(final_size, final_size)])
    resized_feature_maps = transform(image=new_matrix)["image"]

    os.makedirs(save_base_dir, exist_ok=True)
    np.save(f'{save_base_dir}/{wsi_name}.npy', resized_feature_maps)


def zscore_normalization_and_cut(data, tissue_mask_dir, feature_maps_dir,
                                 normalized_save_dir, cut_empty_save_dir,
                                 final_save_dir, is_normalize, final_size):

    # Apply z-score normalization to feature maps and process tissue regions.

    os.makedirs(normalized_save_dir, exist_ok=True)
    os.makedirs(cut_empty_save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)
    finished_wsi = glob.glob(f'{final_save_dir}/*')

    # Calculate normalization parameters if enabled
    if is_normalize:
        use_area_set = None
        for path in tqdm(data):
            if any(path in s for s in finished_wsi):
                print(f'{path} has finished!')
                continue

            wsi_name = path.replace('.npy', '.svs')
            mask_path = f'{tissue_mask_dir}/{wsi_name}/{wsi_name}_mask_use.png'
            tissue_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            feature_maps = np.load(f'{feature_maps_dir}/{path}')

            # Handle NaN values
            if np.isnan(feature_maps).any():
                feature_maps = np.nan_to_num(feature_maps)

            # Resize mask to match feature map dimensions
            high, weight, _ = feature_maps.shape
            tissue_mask = cv.resize(tissue_mask, (weight, high), interpolation=cv.INTER_NEAREST)

            # Accumulate tissue regions for stats calculation
            use_area = feature_maps[tissue_mask != 0]
            use_area_set = use_area if use_area_set is None else np.vstack((use_area_set, use_area))

        means = np.mean(use_area_set, axis=0)
        stds = np.std(use_area_set, axis=0)

    # Process each WSI
    for path in data:
        if any(path in s for s in finished_wsi):
            continue

        wsi_name = path.replace('.npy', '')
        mask_path = f'{tissue_mask_dir}/{wsi_name}.svs/{wsi_name}.svs_mask_use.png'
        tissue_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        try:
            feature_maps = np.load(f'{feature_maps_dir}/{path}')
        except Exception:
            continue

        # Resize and normalize
        high, weight, channel_num = feature_maps.shape
        tissue_mask = cv.resize(tissue_mask, (weight, high), interpolation=cv.INTER_NEAREST)

        if is_normalize:
            for c in range(channel_num):
                temp = feature_maps[:, :, c]
                temp[tissue_mask != 0] -= means[c]
                if stds[c] > 0.0001:  # Avoid division by near-zero
                    temp[tissue_mask != 0] /= stds[c]
                feature_maps[:, :, c] = temp

            # Final NaN check
            if np.isnan(feature_maps).any():
                feature_maps = np.nan_to_num(feature_maps)

            np.save(f'{normalized_save_dir}/{wsi_name}.npy', feature_maps)

        cut_empty_and_resize(feature_maps, tissue_mask, wsi_name,
                             cut_empty_save_dir, final_save_dir, final_size)


def maxmin_normalization_and_cut(data, tissue_mask_dir, feature_maps_dir,
                                 normalized_save_dir, cut_empty_save_dir,
                                 final_save_dir, is_normalize, channel_num):

    os.makedirs(normalized_save_dir, exist_ok=True)
    os.makedirs(cut_empty_save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)
    finished_wsi = glob.glob(f'{final_save_dir}/*')

    # Calculate min/max values across dataset
    max_values = np.zeros(channel_num)
    min_values = np.zeros(channel_num) + 99999

    for path in tqdm(data):
        wsi_name = path.replace('.npy', '.svs')
        mask_path = f'{tissue_mask_dir}/{wsi_name}/{wsi_name}_mask_use.png'
        tissue_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        feature_maps = np.load(f'{feature_maps_dir}/{path}')

        # Handle NaN values
        if np.isnan(feature_maps).any():
            print(f'{wsi_name} has NaN values before normalization')
            feature_maps = np.nan_to_num(feature_maps)

        # Find min/max in tissue regions
        high, weight, _ = feature_maps.shape
        tissue_mask = cv.resize(tissue_mask, (weight, high), interpolation=cv.INTER_NEAREST)
        use_area = feature_maps[tissue_mask != 0]
        max_arr = np.max(use_area, axis=0)
        min_arr = np.min(use_area, axis=0)

        # Update global min/max
        max_values = np.maximum(max_values, max_arr)
        min_values = np.minimum(min_values, min_arr)

    # Normalize and process each WSI
    for path in data:
        if any(path in s for s in finished_wsi):
            continue

        wsi_name = path.replace('.npy', '')
        mask_path = f'{tissue_mask_dir}/{wsi_name}.svs/{wsi_name}.svs_mask_use.png'
        tissue_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        feature_maps = np.load(f'{feature_maps_dir}/{path}')

        high, weight, _ = feature_maps.shape
        tissue_mask = cv.resize(tissue_mask, (weight, high), interpolation=cv.INTER_NEAREST)

        if is_normalize:
            for c in range(channel_num):
                temp = feature_maps[:, :, c]
                temp[tissue_mask != 0] -= min_values[c]

                # Scale to [0,1] range
                scale_factor = 0 if max_values[c] == min_values[c] else 1 / (max_values[c] - min_values[c])
                temp[tissue_mask != 0] *= scale_factor

                feature_maps[:, :, c] = temp

            # Final NaN check
            if np.isnan(feature_maps).any():
                feature_maps = np.nan_to_num(feature_maps)

            np.save(f'{normalized_save_dir}/{wsi_name}.npy', feature_maps)

        cut_empty_and_resize(feature_maps, tissue_mask, wsi_name,
                             cut_empty_save_dir, final_save_dir, 256)


def dataset_preprcess(is_normalize, test_data_file_dir, base_dir,
                                 data_set_name, channel_num, nor_method, final_size):
    """
    Process dataset with specified normalization.
    """
    with open(f'{test_data_file_dir}/all_data.json', 'r', encoding='utf-8-sig') as f:
        data_set = json.load(f)

    if not is_normalize:
        nor_method = 'initial'

    # Set up paths based on feature type
    if channel_num == 40:
        concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/hand-crafted/texture_feature_maps'
    elif channel_num == 8:
        concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/tissue_type/tissue_type_maps'
    elif channel_num == 120:
        concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/hand-crafted/nucleus_feature_maps'
    else:
        concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/initial/concat_feature_maps'

    # Set up output directories
    normalized_save_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/{nor_method}/normalized_feature_maps'
    cut_empty_save_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/{nor_method}/cut_empty_feature_maps'
    final_save_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/{nor_method}/final_feature_maps/{final_size}'
    tissue_mask_dir = f'{base_dir}/{data_set_name}/processed_data/histoqc'

    # Apply selected normalization method
    if nor_method == 'maxmin':
        maxmin_normalization_and_cut(data_set, tissue_mask_dir, concat_feature_maps_dir,
                                     normalized_save_dir, cut_empty_save_dir, final_save_dir,
                                     is_normalize, channel_num)
    else:  # zscore or initial
        zscore_normalization_and_cut(data_set, tissue_mask_dir, concat_feature_maps_dir,
                                     normalized_save_dir, cut_empty_save_dir, final_save_dir,
                                     is_normalize, final_size)


base_dir='HCC_path'
data_set_name='TCGA'
test_data_file_dir=f'{base_dir}/{data_set_name}'
channel_num=168
is_normalize=False
nor_method = 'maxmin'
final_size = 256
dataset_preprcess(is_normalize, test_data_file_dir, base_dir,
                           data_set_name, channel_num, nor_method, final_size)