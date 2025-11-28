import time
import os
import numpy as np
import torch

from utils.model import Network
from utils.normalization import normalize_range, un_normalize_range


def phantom_glu_inference(glu_nn_path, acquired_data_glu, array_shape, device):
    c_acq_data, w_acq_data = array_shape
    # load min-max of dataset
    mt_min_max_data = np.load(os.path.join(os.path.dirname(glu_nn_path), 'min_max_vals.npz'))
    # Access each array by its key
    min_param_tensor_0 = torch.tensor(mt_min_max_data['min_param'])
    max_param_tensor_0 = torch.tensor(mt_min_max_data['max_param'])
    min_water_t1t2_tensor_0 = torch.tensor(mt_min_max_data['min_water_t1t2'])
    max_water_t1t2_tensor_0 = torch.tensor(mt_min_max_data['max_water_t1t2'])

    # load model
    glu_reco_net = Network(30, add_iter=0, output_dim=4).to(device)
    model_weights = torch.load(glu_nn_path, map_location=torch.device(device))
    glu_reco_net.load_state_dict(model_weights)

    # adding the water_t1t2 input as two additional elements in the noised_sig vector
    acquired_data = acquired_data_glu.to(device).float()

    # evaluate
    glu_reco_net.eval()
    t0 = time.time()
    prediction = glu_reco_net(acquired_data.float())
    print(f"Phantom Glutamate prediction took {1000 * (time.time() - t0):.5f} ms")

    # Un-normalizing to go back to physical units
    min_param_tensor = torch.hstack((min_param_tensor_0, min_water_t1t2_tensor_0))
    max_param_tensor = torch.hstack((max_param_tensor_0, max_water_t1t2_tensor_0))
    prediction = un_normalize_range(prediction, original_min=min_param_tensor.to(device),
                                    original_max=max_param_tensor.to(device), new_min=0, new_max=1)

    quant_maps_glu = {}

    # Reshaping back to the image dimension
    quant_maps_glu['fs'] = prediction.cpu().detach().numpy()[:, 0]
    quant_maps_glu['fs'] = quant_maps_glu['fs'].T
    quant_maps_glu['fs'] = np.reshape(quant_maps_glu['fs'], (c_acq_data, w_acq_data), order='F')

    quant_maps_glu['ksw'] = prediction.cpu().detach().numpy()[:, 1]
    quant_maps_glu['ksw'] = quant_maps_glu['ksw'].T
    quant_maps_glu['ksw'] = np.reshape(quant_maps_glu['ksw'], (c_acq_data, w_acq_data), order='F')

    return quant_maps_glu


def mt_inference(mt_nn_path, prepared_array, array_shape, device):
    c_acq_data, w_acq_data = array_shape
    # load min-max of dataset
    mt_min_max_data = np.load(os.path.join(os.path.dirname(mt_nn_path), 'min_max_vals.npz'))
    # Access each array by its key
    min_param_tensor_0 = torch.tensor(mt_min_max_data['min_param'])
    max_param_tensor_0 = torch.tensor(mt_min_max_data['max_param'])
    min_water_t1t2_tensor_0 = torch.tensor(mt_min_max_data['min_water_t1t2'])
    max_water_t1t2_tensor_0 = torch.tensor(mt_min_max_data['max_water_t1t2'])

    # load model
    mt_reco_net = Network(31, add_iter=2).to(device)
    model_weights = torch.load(mt_nn_path, map_location=torch.device(device))
    mt_reco_net.load_state_dict(model_weights)

    # Normalizing according to dict water_t1t2 min and max values
    input_water_t1t2_acquired = torch.hstack((prepared_array['t1_map'], prepared_array['t2_map']))
    input_water_t1t2_acquired = normalize_range(original_array=input_water_t1t2_acquired,
                                                original_min=min_water_t1t2_tensor_0,
                                                original_max=max_water_t1t2_tensor_0, new_min=0, new_max=1).to(device)

    # adding the water_t1t2 input as two additional elements in the noised_sig vector
    acquired_data = torch.hstack((input_water_t1t2_acquired, prepared_array['raw_mt'])).to(device).float()

    # evaluate
    mt_reco_net.eval()
    t0 = time.time()
    prediction = mt_reco_net(acquired_data.float())
    print(f"MT prediction took {1000 * (time.time() - t0):.5f} ms")

    # Un-normalizing to go back to physical units
    prediction = un_normalize_range(prediction, original_min=min_param_tensor_0.to(device),
                                    original_max=max_param_tensor_0.to(device), new_min=0, new_max=1)

    quant_maps_mt = {}

    # Reshaping back to the image dimension
    quant_maps_mt['fs'] = prediction.cpu().detach().numpy()[:, 0]
    quant_maps_mt['fs'] = quant_maps_mt['fs'].T
    quant_maps_mt['fs'] = np.reshape(quant_maps_mt['fs'], (c_acq_data, w_acq_data), order='F')

    quant_maps_mt['ksw'] = prediction.cpu().detach().numpy()[:, 1]
    quant_maps_mt['ksw'] = quant_maps_mt['ksw'].T
    quant_maps_mt['ksw'] = np.reshape(quant_maps_mt['ksw'], (c_acq_data, w_acq_data), order='F')

    input_mt_param = prediction.cpu().detach().numpy()
    input_mt_param = torch.from_numpy(input_mt_param)

    return input_mt_param, quant_maps_mt

def rnoe_inference(rnoe_nn_path, prepared_array, input_mt_param, array_shape, device):
    c_acq_data, w_acq_data = array_shape
    # load min-max of dataset
    min_max_data = np.load(os.path.join(os.path.dirname(rnoe_nn_path), 'min_max_vals.npz'))
    # Target:
    min_glu_param_tensor_1 = torch.tensor(min_max_data['min_param'])
    max_glu_param_tensor_1 = torch.tensor(min_max_data['max_param'])
    # Input:
    min_water_t1t2_tensor_1 = torch.tensor(min_max_data['min_water_t1t2'])
    max_water_t1t2_tensor_1 = torch.tensor(min_max_data['max_water_t1t2'])
    min_mt_param_tensor_1 = torch.tensor(min_max_data['min_mt_param'])
    max_mt_param_tensor_1 = torch.tensor(min_max_data['max_mt_param'])

    # load model
    reco_net = Network(31, add_iter=4, output_dim=2, n_hidden=2).to(device)
    model_weights = torch.load(rnoe_nn_path, map_location=torch.device(device))
    reco_net.load_state_dict(model_weights)

    # Normalizing according to dict water_t1t2 min and max values
    input_water_t1t2_acquired = torch.hstack((prepared_array['t1_map'], prepared_array['t2_map']))
    input_water_t1t2_acquired = normalize_range(original_array=input_water_t1t2_acquired,
                                                original_min=min_water_t1t2_tensor_1,
                                                original_max=max_water_t1t2_tensor_1, new_min=0, new_max=1).to(device)

    input_mt_param = normalize_range(original_array=input_mt_param,
                                     original_min=min_mt_param_tensor_1,
                                     original_max=max_mt_param_tensor_1, new_min=0, new_max=1).to(device)

    # concat input
    acquired_data = torch.hstack((input_mt_param, input_water_t1t2_acquired, prepared_array['raw_rnoe'])).to(device).float()

    # evaluate
    reco_net.eval()
    t0 = time.time()
    prediction = reco_net(acquired_data.float())
    print(f"rNOE prediction took {1000 * (time.time() - t0):.5f} ms")

    # Un-normalizing to go back to physical units
    prediction = un_normalize_range(prediction, original_min=min_glu_param_tensor_1.to(device),
                                    original_max=max_glu_param_tensor_1.to(device), new_min=0, new_max=1)

    quant_maps_noe = {}

    # Reshaping back to the image dimension
    quant_maps_noe['fs'] = prediction.cpu().detach().numpy()[:, 0]
    quant_maps_noe['fs'] = quant_maps_noe['fs'].T
    quant_maps_noe['fs'] = np.reshape(quant_maps_noe['fs'], (c_acq_data, w_acq_data), order='F')

    quant_maps_noe['ksw'] = prediction.cpu().detach().numpy()[:, 1]
    quant_maps_noe['ksw'] = quant_maps_noe['ksw'].T
    quant_maps_noe['ksw'] = np.reshape(quant_maps_noe['ksw'], (c_acq_data, w_acq_data), order='F')

    return quant_maps_noe

def amide_glu_inference(amide_glu_nn_path, prepared_array, input_mt_param, array_shape, device):
    c_acq_data, w_acq_data = array_shape
    # load min-max of dataset
    ga_min_max_data = np.load(os.path.join(os.path.dirname(amide_glu_nn_path), 'min_max_vals.npz'))
    # Target:
    min_glu_param_tensor_1 = torch.tensor(ga_min_max_data['min_param'])
    max_glu_param_tensor_1 = torch.tensor(ga_min_max_data['max_param'])
    min_amide_param_tensor_1 = torch.tensor(ga_min_max_data['min_amide_param'])
    max_amide_param_tensor_1 = torch.tensor(ga_min_max_data['max_amide_param'])
    # Input:
    min_water_t1t2_tensor_1 = torch.tensor(ga_min_max_data['min_water_t1t2'])
    max_water_t1t2_tensor_1 = torch.tensor(ga_min_max_data['max_water_t1t2'])
    min_mt_param_tensor_1 = torch.tensor(ga_min_max_data['min_mt_param'])
    max_mt_param_tensor_1 = torch.tensor(ga_min_max_data['max_mt_param'])

    # load model
    reco_net = Network(62, add_iter=4, output_dim=4, n_hidden=4).to(device)
    model_weights = torch.load(amide_glu_nn_path, map_location=torch.device(device))
    reco_net.load_state_dict(model_weights)

    # Normalizing according to dict water_t1t2 min and max values
    input_water_t1t2_acquired = torch.hstack((prepared_array['t1_map'], prepared_array['t2_map']))
    input_water_t1t2_acquired = normalize_range(original_array=input_water_t1t2_acquired,
                                                original_min=min_water_t1t2_tensor_1,
                                                original_max=max_water_t1t2_tensor_1, new_min=0, new_max=1).to(device)

    input_mt_param = normalize_range(original_array=input_mt_param,
                                     original_min=min_mt_param_tensor_1,
                                     original_max=max_mt_param_tensor_1, new_min=0, new_max=1).to(device)

    # concat input
    acquired_data = torch.hstack((input_mt_param, input_water_t1t2_acquired, prepared_array['raw_glu'], prepared_array['raw_amide'])).to(
        device).float()

    # evaluate
    reco_net.eval()
    t0 = time.time()
    prediction = reco_net(acquired_data.float())
    print(f"Amide and Glutamate prediction took {1000 * (time.time() - t0):.5f} ms")

    # Un-normalizing to go back to physical units
    prediction = un_normalize_range(prediction,
                                    original_min=torch.hstack((min_glu_param_tensor_1, min_amide_param_tensor_1)).to(
                                        device),
                                    original_max=torch.hstack((max_glu_param_tensor_1, max_amide_param_tensor_1)).to(
                                        device), new_min=0, new_max=1)

    # Reshaping back to the image dimension
    quant_maps_amide = {}
    quant_maps_amide['fs'] = prediction.cpu().detach().numpy()[:, 2]
    quant_maps_amide['fs'] = quant_maps_amide['fs'].T
    quant_maps_amide['fs'] = np.reshape(quant_maps_amide['fs'], (c_acq_data, w_acq_data), order='F')

    quant_maps_amide['ksw'] = prediction.cpu().detach().numpy()[:, 3]
    quant_maps_amide['ksw'] = quant_maps_amide['ksw'].T
    quant_maps_amide['ksw'] = np.reshape(quant_maps_amide['ksw'], (c_acq_data, w_acq_data), order='F')

    # Reshaping back to the image dimension
    quant_maps_glu = {}
    quant_maps_glu['fs'] = prediction.cpu().detach().numpy()[:, 0]
    quant_maps_glu['fs'] = quant_maps_glu['fs'].T
    quant_maps_glu['fs'] = np.reshape(quant_maps_glu['fs'], (c_acq_data, w_acq_data), order='F')

    quant_maps_glu['ksw'] = prediction.cpu().detach().numpy()[:, 1]
    quant_maps_glu['ksw'] = quant_maps_glu['ksw'].T
    quant_maps_glu['ksw'] = np.reshape(quant_maps_glu['ksw'], (c_acq_data, w_acq_data), order='F')

    return quant_maps_amide, quant_maps_glu

