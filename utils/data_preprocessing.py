import numpy as np

import torch
from torch.autograd import Variable


def img_cropper(mask_cen, img_to_crop, highres_flag=False):
    r_c, c_c = int(mask_cen[0]), int(mask_cen[1])
    bb_r_h = 15
    bb_c_w = 25

    if len(img_to_crop.shape) == 3:
        # If the image has 3 channels (e.g. MRF)
        cropped_img = img_to_crop[r_c-bb_r_h:r_c+bb_r_h, c_c-bb_c_w:c_c+bb_c_w, :]
    elif len(img_to_crop.shape) == 2:
        # If the image has 2 channels (e.g., T1, T2, mask, highres)
        if highres_flag:
            r_c, c_c = 2 * r_c, 2 * c_c
            bb_r_h = 2 * bb_r_h
            bb_c_w = 2 * bb_c_w
            cropped_img = img_to_crop[r_c - bb_r_h:r_c + bb_r_h, c_c - bb_c_w:c_c + bb_c_w]
        else:
            cropped_img = img_to_crop[r_c-bb_r_h:r_c+bb_r_h, c_c-bb_c_w:c_c+bb_c_w]

    return cropped_img


def load_mrf(acquired_data, dtype, device):
    [cha_n, c_acq_data, w_acq_data] = np.shape(acquired_data)

    # Reshaping the acquired data to the shape expected by the NN (e.g. 31 x ... )
    acquired_data = np.reshape(acquired_data, (cha_n, c_acq_data * w_acq_data), order='F')

    # 2-norm normalization
    acquired_data = acquired_data / np.linalg.norm(acquired_data, axis=0, ord=2)

    # Transposing for compatibility with the NN - now each row is a trajectory
    acquired_data = acquired_data.T

    # Converting to tensor
    acquired_data = Variable(torch.from_numpy(acquired_data).type(dtype), requires_grad=False).to(device)

    return acquired_data


def phantom_data_preparation(glu_scan, dtype, device):
    # Loading the MRF acquisitions
    prepared_glu_scan = load_mrf(glu_scan, dtype, device)
    return prepared_glu_scan


def data_preparation(scan_array, dtype, device):
    prepared_array = {}

    # Loading the MRF acquisitions
    prepared_array['raw_mt'] = load_mrf(scan_array['raw_mt'], dtype, device)
    prepared_array['raw_rnoe'] = load_mrf(scan_array['raw_rnoe'], dtype, device)
    prepared_array['raw_amide'] = load_mrf(scan_array['raw_amide'], dtype, device)
    prepared_array['raw_glu'] = load_mrf(scan_array['raw_glu'], dtype, device)

    # Loading t1, t2 maps
    t1 = scan_array['t1_map']
    t2 = scan_array['t2_map']
    [c_acq_data, w_acq_data] = np.shape(t1)

    # Reshaping the acquired t1 data to the shape expected by the NN (e.g. 1 x ... )
    acquired_map_t1w_orig = t1.astype(np.float32) / 1000
    acquired_map_t1w = np.reshape(acquired_map_t1w_orig, (1, c_acq_data * w_acq_data), order='F').T
    prepared_array['t1_map'] = torch.from_numpy(acquired_map_t1w)

    # Reshaping the acquired t2 data to the shape expected by the NN (e.g. 1 x ... )
    acquired_map_t2w_orig = t2.astype(np.float32) / 1000
    acquired_map_t2w = np.reshape(acquired_map_t2w_orig, (1, c_acq_data * w_acq_data), order='F').T
    prepared_array['t2_map'] = torch.from_numpy(acquired_map_t2w)

    return prepared_array