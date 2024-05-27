import os
from typing import Dict, Tuple
from tqdm import tqdm

import h5py
from joblib import load, dump

import pandas as pd
import numpy as np


def read_multicoil(file_name):
    """ 
    return k-space, reconstructed images - esc and rss 
    """
    f = h5py.File(file_name, 'r')
    kspace = f['kspace'][:]
    recon_rss = f['reconstruction_rss'][:]

    return dict(kspace=kspace, recon_rss=recon_rss)


def processed_kspace(file_name, kspace_shape: Tuple[int, int]) -> Dict:
    """ 
    zero-fill k-space to kspace_shape 
    indices where k-space sampling starts and ends    
    """
    data_dict = read_multicoil(file_name=file_name)
    kspace = data_dict['kspace']

    # zero-fill k-space
    n_slices, n_channels, n_rows, n_cols = kspace.shape

    if n_rows == kspace_shape[0] and n_cols == kspace_shape[1]:

        # k-space sampled rows
        sampled_indices = (np.abs(kspace[0]).sum(axis=0) > 0)
        left_index, right_index = np.where(sampled_indices)[0][[0, -1]]

        # added start and end indices
        data_dict['sampled_indices'] = [left_index, right_index]
        return data_dict

    elif n_cols > 400:
        return None

    else:
        n_rows_to_fill = kspace_shape[1] - n_cols

        zero_filled_kspace = np.zeros(
            (n_slices, n_channels, kspace_shape[0], kspace_shape[1])) + 1j * 0.0

        zero_filled_kspace[:, :, :, n_rows_to_fill //
                           2: -n_rows_to_fill//2] = kspace

        # k-space sampled rows
        sampled_indices = (np.abs(zero_filled_kspace[0]).sum(axis=0) > 0)
        left_index, right_index = np.where(sampled_indices)[0][[0, -1]]

        # added start and end indices
        data_dict['sampled_indices'] = [left_index, right_index]
        data_dict['kspace'] = zero_filled_kspace

        return data_dict


def add_multicoil(file_name, save_root, kspace_shape):

    data_dict = processed_kspace(file_name=file_name,
                                 kspace_shape=kspace_shape)

    if data_dict is None:
        return None

    volume_id = file_name.split("/")[-1].replace('.h5', '')
    volume_path = os.path.join(save_root, volume_id)

    os.makedirs(volume_path, exist_ok = True)
    for slice_id in range(data_dict['kspace'].shape[0]):
        # get slice path
        slice_filename = f'knee_singlecoil_{volume_id}_{slice_id}.h5'
        slice_path = os.path.join(volume_path, slice_filename)

        hf =  h5py.File(slice_path, mode='w')
        hf.create_dataset(
            'mc_kspace', data=data_dict['kspace'][slice_id])
        hf.create_dataset(
            'recon_rss', data=data_dict['recon_rss'][slice_id])


def main():
    path = '../knee/'
    train_paths = os.listdir(os.path.join(path, 'multicoil_train'))
    train_paths = [os.path.join(path, 'multicoil_train', x)
                   for x in train_paths]

    val_paths = os.listdir(os.path.join(path, 'multicoil_val'))
    val_paths = [os.path.join(path, 'multicoil_val', x) for x in val_paths]
    paths = train_paths + val_paths

    save_root = '../knee/'
    kspace_shape = (640, 400)

    for file_name in tqdm(paths):
        add_multicoil(file_name=file_name, save_root=save_root,
                      kspace_shape=kspace_shape)


if __name__ == '__main__':
    main()
