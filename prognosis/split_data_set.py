import json
import glob
import os
import pandas as pd
import random


def clean_tcga_paths(segmentation_path_list):

    metadata_df = pd.read_csv('HCC_path/TCGA/TCGA.csv')
    cleaned_paths = []
    metadata_df['WSIs'] = metadata_df['WSIs'].astype(str)

    for seg_path in segmentation_path_list:
        case_id = os.path.basename(seg_path).split('.')[0][0:23]
        try:
            # Check if case exists in metadata
            pd_index = metadata_df[metadata_df['WSIs'].isin([case_id])].index.values[0]
            cleaned_paths.append(seg_path)
        except:
            print(f"Case ID not found: {case_id}")
    return cleaned_paths


def create_k_fold_splits(path_list, num_folds):

    k_fold_splits = []
    total_cases = len(path_list)
    fold_size = int(total_cases / num_folds)

    for fold_idx in range(num_folds):
        # Create test set for current fold
        if fold_idx < num_folds - 1:
            test_set = path_list[fold_idx * fold_size:(fold_idx + 1) * fold_size]
        else:
            test_set = path_list[fold_idx * fold_size:]

        # Create corresponding train set
        train_set = [path for path in path_list if path not in test_set]
        k_fold_splits.append([train_set, test_set])

    return k_fold_splits


if __name__ == '__main__':

    feature_maps_dir = 'HCC_path/CohortLIHC/processed_data/feature_maps/concat_feature_maps/168d/initial/final_feature_maps/256'
    output_dir = 'HCC_path/TCGA/'
    num_folds = 10
    random_seed = 2024

    # Step 1: Get all feature map paths
    feature_map_paths = []
    for path in glob.glob(os.path.join(feature_maps_dir, '*')):
        wsi_name = os.path.basename(path)
        full_path = os.path.join(feature_maps_dir, wsi_name)
        feature_map_paths.append(full_path)

    # Step 2: Clean paths and sort
    cleaned_feature_map_paths = clean_tcga_paths(feature_map_paths)
    cleaned_feature_map_paths.sort()

    # Step 3: Extract case IDs and save to all_data.json
    all_case_ids = [os.path.basename(path) for path in cleaned_feature_map_paths]

    # Save all case IDs to JSON file
    all_data_path = os.path.join(output_dir, 'all_data.json')
    with open(all_data_path, 'w') as f:
        json.dump(all_case_ids, f)
    print(f"Saved {len(all_case_ids)} cases to {all_data_path}")

    # Step 4: Shuffle case IDs with fixed seed for reproducibility
    random.seed(random_seed)
    random.shuffle(all_case_ids)

    # Step 5: Create 10-fold splits
    cross_val_splits = create_k_fold_splits(all_case_ids, num_folds=num_folds)

    # Step 6: Save each fold to separate JSON files
    for fold_idx, (train_data, test_data) in enumerate(cross_val_splits):
        fold_data = {
            'train_data': train_data,
            'test_data': test_data
        }
        output_path = os.path.join(output_dir, f'split_data_fold_{fold_idx}.json')
        with open(output_path, 'w') as file:
            json.dump(fold_data, file)
        print(f"Saved fold {fold_idx} with {len(train_data)} train and {len(test_data)} test cases")