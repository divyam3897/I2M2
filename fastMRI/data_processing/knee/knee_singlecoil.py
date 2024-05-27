import os
from typing import Dict, Tuple
from tqdm import tqdm

import h5py
from joblib import load, dump

import pandas as pd
import numpy as np


def read_singlecoil(file_name):
    """ 
    return k-space, reconstructed images - esc and rss 
    """
    f = h5py.File(file_name, 'r')

    kspace = f['kspace'][:]
    recon_rss = f['reconstruction_rss'][:]
    recon_esc = f['reconstruction_esc'][:]

    return dict(kspace=kspace, recon_rss=recon_rss, recon_esc=recon_esc)


def processed_kspace(file_name, kspace_shape: Tuple[int, int]) -> Dict:
    """ 
    zero-fill k-space to kspace_shape 
    indices where k-space sampling starts and ends    
    """
    data_dict = read_singlecoil(file_name=file_name)
    kspace = data_dict['kspace']

    # zero-fill k-space
    n_slices, n_rows, n_cols = kspace.shape

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
            (n_slices, kspace_shape[0], kspace_shape[1])) + 1j * 0.0
        zero_filled_kspace[:, :, n_rows_to_fill //
                           2: -n_rows_to_fill//2] = kspace

        # k-space sampled rows
        sampled_indices = (np.abs(zero_filled_kspace[0]).sum(axis=0) > 0)
        left_index, right_index = np.where(sampled_indices)[0][[0, -1]]

        # added start and end indices
        data_dict['sampled_indices'] = [left_index, right_index]
        data_dict['kspace'] = zero_filled_kspace

        return data_dict


def save_single_coil(file_name: str,
                     annotations: pd.DataFrame,
                     save_root: str,
                     kspace_shape: Tuple[int, int],
                     meta_data: Dict):
    data_dict = processed_kspace(
        file_name=file_name, kspace_shape=kspace_shape)

    if data_dict is None:
        return None

    volume_id = file_name.split("/")[-1].replace(".h5", "")
    volume_path = os.path.join(save_root, volume_id)

    df = annotations[annotations.file == volume_id]
    slices_with_labels = np.unique(list(df.slice))

    # make volume folder
    os.makedirs(volume_path)
    for slice_id in range(data_dict['kspace'].shape[0]):

        # make slice path
        slice_filename = f'knee_singlecoil_{volume_id}_{slice_id}.h5'

        slice_path = os.path.join(volume_path, slice_filename)

        hf = h5py.File(slice_path, 'w')
        hf.create_dataset('sc_kspace', data=data_dict['kspace'][slice_id])
        hf.create_dataset(
            'recon_esc', data=data_dict['recon_esc'][slice_id])
        hf.create_dataset(
            'recon_rss', data=data_dict['recon_rss'][slice_id])
        hf.create_dataset('sampled_indices',
                          data=data_dict['sampled_indices'])

        # get labels
        if slice_id not in slices_with_labels:
            # no labels for this slice
            labels = ['None']

        elif slice_id in slices_with_labels:
            labels = list(df[df.slice == slice_id].label)

        hf.create_dataset('labels', data=labels)
        hf.close()

        meta_data[f'knee_{volume_id}_{slice_id}'] = dict(volume_id=volume_id,
                                                         slice_id=slice_id,
                                                         labels=labels,
                                                         shape=data_dict['kspace'].shape,
                                                         dataset='knee',
                                                         data_type='original')


def main():
    # k-space shape after zero-filling in phase encoding direction
    kspace_shape = (640, 400)

    path = '../knee/'
    annotations = pd.read_csv('../Annotations/knee.csv')
    save_root = '../knee/'

    train_paths = os.listdir(os.path.join(path, 'singlecoil_train'))
    train_paths = [os.path.join(path, 'singlecoil_train', x)
                   for x in train_paths]

    val_paths = os.listdir(os.path.join(path, 'singlecoil_val'))
    val_paths = [os.path.join(path, 'singlecoil_val', x) for x in val_paths]

    paths = train_paths + val_paths
    meta_data = {}
    for i, file_name in enumerate(tqdm(paths)):
        save_single_coil(file_name=file_name,
                         annotations=annotations,
                         save_root=save_root,
                         kspace_shape=kspace_shape,
                         meta_data=meta_data)

    meta_data = pd.DataFrame(meta_data).T
    dump(meta_data, os.path.join(save_root, 'meta_data.p'))


if __name__ == '__main__':
    main()
