import os
import h5py
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    knee_data_root =  "../knee/"
    train_path = knee_data_root + "singlecoil_train/"
    val_path = knee_data_root + "singlecoil_val/"
    test_path = knee_data_root + "singlecoil_test_v2/"
    
    
    ## Read metadata and split data
    df_knee = pd.read_csv('../knee.csv')
    df_knee_file = pd.read_csv('../Annotations/knee_file_list.csv', header=None)


    ## Total number of volumes for which classification is available
    df_knee_file.shape
    classified_vols = list(set(df_knee_file[0].values))


    ## Total number of volumes
    train_vols, val_vols, test_vols = os.listdir(train_path), os.listdir(val_path), os.listdir(test_path)
    all_volumes = train_vols + val_vols ## cannot consider test as no reconstruction is available for it
    all_volumes = [vol.replace('.h5', '') for vol in all_volumes]

    ## volumes that are not classified
    nonclassified_vols = list(set(all_volumes) - set(classified_vols))

    print("All volumes: ",len(all_volumes))
    print("Classified volumes: ",len(classified_vols))
    print("Non-Classified volumes: ",len(nonclassified_vols))
    train_class, val_class = train_test_split(classified_vols, test_size=0.3, random_state=0)
    test_class, val_class = train_test_split(val_class, test_size=0.5, random_state=0)

    def get_data_split(file_name):
        if file_name in set(train_class):
            return "train_class"
        elif file_name in set(val_class):
            return "val_class"
        elif file_name in set(test_class):
            return "test_class"
        return None


    def get_labels(file_name):
        if file_name in file_label_dict:
            return list(set(file_label_dict.get(file_name)))
        return [None]

    df_knee_file_slice_level = df_knee.groupby(['file', 'slice'])['label'].apply(list).reset_index()


    df_knee_file_slice_level['key'] = df_knee_file_slice_level['file'] + "_" + df_knee_file_slice_level['slice'].astype(str)
    file_label_dict = dict(zip(df_knee_file_slice_level.key, df_knee_file_slice_level.label))

    index, volume_ids, slice_ids, shapes = [], [], [], []
    data_split, labels, locations = [], [], []


    root_path = '../knee/'
    folder_list = os.listdir(root_path)

    for folder in tqdm.tqdm(folder_list):
        files_path = os.path.join(root_path, folder)
        files_in_folder = os.listdir(files_path)
    #     print(folder)
        for file_in_folder in files_in_folder:
            # file_in_folder contains .h5
            file_path = os.path.join(files_path, file_in_folder)
            file = h5py.File(file_path)
            kspace = file['sc_kspace']

            vol_name = folder
            slice_id = file_in_folder.split('.')[0].split('_')[-1]
            shape = kspace.shape

            data_split_val = get_data_split(vol_name)
            label_val = get_labels('{}_{}'.format(vol_name, slice_id))

            index_str = 'knee_{}'.format(file_in_folder.split('.')[0])

            index.append(index_str)
            volume_ids.append(vol_name)
            slice_ids.append(slice_id)
            shapes.append(shape)
            data_split.append(data_split_val)
            labels.append(label_val)
            locations.append(file_path)
            
    metadata_df = pd.DataFrame()
    metadata_df['index'] = index
    metadata_df['volume_id'] = volume_ids
    metadata_df['slice_id'] = slice_ids
    metadata_df['labels'] = labels
    metadata_df['shape'] = shapes
    metadata_df['dataset'] = 'knee'
    metadata_df['data_type'] = 'orignial'
    metadata_df['data_split'] = data_split
    metadata_df['location'] = locations
            
    metadata_df.to_csv(root_path + 'metadata_knee.csv')
    
if __name__ == '__main__':
    main()
