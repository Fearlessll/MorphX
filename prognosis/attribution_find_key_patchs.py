import glob
import json
import openslide
from captum.attr import IntegratedGradients, NoiseTunnel
from prognosis.Networks.resnet import resnet10
from data_loaders import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_Atrrmap(file_path, model_path, channel_num):

    # Load model and prepare input
    PATH = model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet10(first_covd_param=[3,2,1], input_channel_num=channel_num,
                        output_use_sigmoid=True, is_attribution=True)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Process input and compute attributions
    input_npy = np.load(file_path)
    if np.sum(np.isnan(input_npy)) != 0:
        input_npy = np.nan_to_num(input_npy)
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            ToTensorV2(),
            ]
        )

    transformed = train_transform(image=input_npy)
    transformed_image = transformed["image"]
    input_tensor = transformed_image.unsqueeze(0).float()
    input_tensor = input_tensor.to(device).type(torch.cuda.FloatTensor)

    ig = IntegratedGradients(model)
    attrmap = ig.attribute(input_tensor, target=None)

    return attrmap


def find_peaks_and_patches(image, num_peaks):

    # Flatten and sort attribution values
    two_d_array = image
    one_d_array = two_d_array.flatten()
    positions = np.unravel_index(np.arange(one_d_array.size), two_d_array.shape)
    rows, cols = positions
    sorted_indices = np.argsort(one_d_array)[::-1]
    sorted_rows = rows[sorted_indices]
    sorted_cols = cols[sorted_indices]
    sorted_one_d_array = one_d_array[sorted_indices]

    return sorted_one_d_array, sorted_rows, sorted_cols,


def get_ori_feature_map_size_arr(attrmap, matrix_path, mask_dir):


    matrix = np.load(matrix_path)
    zeros_arr = np.zeros((matrix.shape[0], matrix.shape[1]))

    # Load and process mask
    wsi_name = os.path.basename(matrix_path).strip('.npy')+'.svs'
    mask_path = os.path.join(mask_dir, wsi_name, wsi_name + '_mask_use.png')
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    high, weight, channel_num = matrix.shape
    mask2 = cv.resize(mask, (weight, high),interpolation = cv.INTER_NEAREST)

    empty = matrix[:,:,1]
    empty[mask2 == 0] = 1
    empty_copy = np.copy(empty)
    matrix[empty == 1] = 0
    matrix[:,:,1] = empty_copy
    #matrix[mask2 == 0] = 0

    coor = np.where(mask2 != 0)
    top = min(coor[0])
    bottom = max(coor[0])
    left = min(coor[1])
    right = max(coor[1])

    # Crop and align attribution map with original coordinates
    is_width_middle_empty = False
    for w in range(left, right):
        if is_width_middle_empty == False and np.sum(mask2[:, w]) == 0:
            start_w = w
            is_width_middle_empty = True
        if is_width_middle_empty == True and np.sum(mask2[:, w]) > 0:
            end_w = w
            break
    is_high_middle_empty = False
    for h in range(top, bottom):
        if is_high_middle_empty == False and np.sum(mask2[h, :]) == 0:
            start_h = h
            is_high_middle_empty = True
        if is_high_middle_empty == True and np.sum(mask2[h, :]) > 0:
            end_h = h
            break
    if is_width_middle_empty == True:
        matrix = np.concatenate([matrix[:, left:start_w], matrix[:, end_w:right]], axis=1)
    else:
        matrix = matrix[:, left:right]
    if is_high_middle_empty == True:
        matrix = np.concatenate([matrix[top:start_h, :], matrix[end_h:bottom, :]], axis=0)
    else:
        matrix = matrix[top:bottom, :]
    height, width, _ = matrix.shape
    new_size = 0

    if height >= width:

        dealt_width = int((height-width)/2)
        ori_value = attrmap[4:-4, dealt_width+4:width+(dealt_width+4)]

        if is_high_middle_empty == True:
            ori_value = np.concatenate([ori_value[:start_h-top, :], np.zeros((end_h-start_h, ori_value.shape[1])), ori_value[start_h-top:, :]], axis=0)
        if is_width_middle_empty == True:
            ori_value = np.concatenate([ori_value[:, :start_w-left], np.zeros((ori_value.shape[0], end_w-start_w)), ori_value[:, start_w-left:]], axis=1)

        zeros_arr[top:bottom, left:right] = ori_value

    else:
        dealt_height = int((width-height)/2)
        ori_value = attrmap[dealt_height+4:height+(dealt_height+4),4:-4]
        if is_high_middle_empty == True:
            ori_value = np.concatenate([ori_value[:start_h - top, :], np.zeros((end_h - start_h, ori_value.shape[1])),
                                        ori_value[start_h - top:, :]], axis=0)
        if is_width_middle_empty == True:
            ori_value = np.concatenate([ori_value[:, :start_w - left], np.zeros((ori_value.shape[0], end_w - start_w)),
                                        ori_value[:, start_w - left:]], axis=1)

        zeros_arr[top:bottom, left:right] = ori_value

    zeros_arr = normalize_array(zeros_arr)
    return zeros_arr, mask


def normalize_array(arr):

    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_min != arr_max:
        normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        normalized_arr = arr

    return normalized_arr

def vis_heat_map(slide, attributions_ig, save_heat_map_dir, wsi_name):

    image = slide.get_thumbnail(slide.level_dimensions[-1])
    heatmap_data = attributions_ig
    image = image.resize((heatmap_data.shape[1], heatmap_data.shape[0]))

    fig, ax = plt.subplots()
    ax.imshow(image)
    heatmap = ax.imshow(heatmap_data, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    plt.savefig(os.path.join(save_heat_map_dir, wsi_name+'.png'), dpi=500)

def get_patch(row, col, peak, slide, save_dir):

    w = col*150
    h = row*150

    kernel_img = slide.read_region((w-300, h-300), 0, (750, 750))
    kernel_img = kernel_img.convert("RGB")
    kernel_img.save(os.path.join(save_dir, f'{round(peak, 5)}_{w}_{h}.png'))
    return True


def attribution_get_key_patch(model_path,wsi_base_dir,mask_dir,seg_dir,all_concat_maps,cut_concat_maps_dir,feature_maps_dir,save_patch_dir,save_heat_map_dir,
                              my_mag = 20,
                              num_peaks = 16,
                              channel_num=168
                              ):

    os.makedirs(save_patch_dir, exist_ok=True)
    os.makedirs(save_heat_map_dir, exist_ok=True)
    finished_wsi_paths = glob.glob(save_patch_dir+'/*')
    for concat_maps_path in all_concat_maps:
        wsi_name = os.path.basename(concat_maps_path).strip('.npy')
        if any(wsi_name in s for s in finished_wsi_paths) == True:
            print(wsi_name+' has finished!')
            continue
        print(wsi_name)
        slide = openslide.open_slide(os.path.join(wsi_base_dir, wsi_name+'.svs'))
        max_mag = int(slide.properties['aperio.AppMag'])

        cut_concat_maps_path = os.path.join(cut_concat_maps_dir, os.path.basename(concat_maps_path))
        try:
            cut_concat_maps = np.load(cut_concat_maps_path)
        except Exception:
            print("load cut_featuremaps error!" + wsi_name)
            continue
        cut_shape = (cut_concat_maps.shape[1], cut_concat_maps.shape[0])

        feature_maps_path = os.path.join(feature_maps_dir, os.path.basename(concat_maps_path))

        # Compute attribution map
        attrmap = get_Atrrmap(feature_maps_path, model_path, channel_num)

        a = attrmap.detach().cpu().numpy()
        a = np.squeeze(a)
        a = abs(a).sum(axis=0)
        a = cv.resize(a, cut_shape, interpolation=cv.INTER_LINEAR)

        # Process and visualize attribution map
        ori_size_attmap, mask = get_ori_feature_map_size_arr(a, concat_maps_path, mask_dir)
        vis_heat_map(slide, ori_size_attmap, save_heat_map_dir, wsi_name)

        os.makedirs(save_heat_map_dir, exist_ok=True)

        # Extract top patches
        sorted_one_d_array, sorted_rows, sorted_cols = find_peaks_and_patches(ori_size_attmap, 1000)
        save_dir = os.path.join(save_patch_dir, wsi_name)
        os.makedirs(save_dir, exist_ok=True)

        image_size = ori_size_attmap.shape
        use_mask = np.zeros(image_size)
        num_patchs = 0

        for i in range(len(sorted_one_d_array)):
            row, col, peak = sorted_rows[i], sorted_cols[i], sorted_one_d_array[i]

            if use_mask[row, col] == 0:
                result = get_patch(row, col, peak, slide, save_dir)
                if result == True:
                    use_mask[row - 2:row + 3, col - 2:col + 3] = 1
                    num_patchs += 1

                if num_patchs >= num_peaks:
                    break


if __name__ == '__main__':

    for num_peaks in [64]:
        for fold, best_index in [(0,19)]:
            for data_set in ['TCGA']:
                my_mag = 20
                channel_num = 168
                base_dir = 'HCC_path'
                data_set_name = data_set
                train_log_dir = 'train_log_TCGA_256_macro_8+40+120_6.6e-4'

                model_path = f'{train_log_dir}/{fold}th/{best_index}.pkl'
                wsi_base_dir = f'{base_dir}/{data_set_name}/raw_data'
                mask_dir = f'{base_dir}/{data_set_name}/processed_data/histoqc'
                if channel_num == 40:
                    concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/hand-crafted/texture_feature_maps'
                elif channel_num == 8:
                    concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/tissue_type/tissue_type_maps'
                elif channel_num == 120:
                    concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/hand-crafted/nucleus_feature_maps'
                else:
                    concat_feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/initial/concat_feature_maps'
                all_concat_maps = glob.glob(concat_feature_maps_dir + '/*')
                data_set_json_path = f'HCC_path/{data_set_name}/all_data.json'
                with open(data_set_json_path, 'r', encoding='utf-8-sig') as f:
                    d_set = json.load(f)
                all_concat_maps = [os.path.join(concat_feature_maps_dir, n) for n in d_set]
                cut_concat_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/initial/cut_empty_feature_maps'
                feature_maps_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/concat_feature_maps/{channel_num}d/initial/final_feature_maps/256'
                save_patch_dir = f'{train_log_dir}/{fold}th_{best_index}_attribution_{num_peaks}_key_patchs_{data_set_name}_new'
                seg_dir = f'{base_dir}/{data_set_name}/processed_data/feature_maps/tissue_type/tissue_type_maps'
                save_heat_map_dir = f'{train_log_dir}/{fold}th_{best_index}_attribution_heat_map_{data_set_name}'
                attribution_get_key_patch(model_path, wsi_base_dir, mask_dir, seg_dir, all_concat_maps, cut_concat_maps_dir,
                                          feature_maps_dir, save_patch_dir, save_heat_map_dir,
                                          my_mag,
                                          num_peaks,
                                          channel_num
                                          )



