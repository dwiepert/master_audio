from pathlib import Path
import random 
from typing import Union, List, Tuple
import tempfile

import pandas as pd

from naip_asr.io import upload_to_gcs

def load_input_data(data_split_root: Union[str, Path], target_labels: List[str], output_dir: Union[str,Path], 
                    cloud:dict, bucket, val_size:int= 50, seed:int=None, uid_col='uid', subject_col='subject') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load labels for each datasplit

    :param data_split_root: path to directory where datasplit csvs are stored
    :param target_labels: list of target label column names
    :param output_dir: path to save a new validation set to if one does not current exist
    :param cloud: dictionary with booleans indicating which directories are cloud based
    :param bucket: cloud bucket
    :param val_size: size of validation set. Set based on whether subject col is in the datasplit csvs or not (if yes, val_size = #participants, else val_size = #files)
    :param seed: seed for random number generation
    :param uid_col: column name for uids
    :param subject_col: column name for participant ids
    :return: train/test/val dataframes
    """
    data_split_root = Path(data_split_root)
    train_path = data_split_root /'train.csv'
    test_path = data_split_root / 'test.csv'

    #get data
    train_df = pd.read_csv(train_path, index_col = uid_col)
    test_df = pd.read_csv(test_path, index_col = uid_col)

    try:
        val_path = data_split_root / 'validation.csv'
        val_df = pd.read_csv(val_path, index_col = uid_col)
        if subject_col in train_df.columns and subject_col in val_df.columns:
            train_df = train_df.loc[~train_df[subject_col].isin(val_df[subject_col].drop_duplicates().to_list())] #double check that no validation speakers are in the train set

    except:
        #randomly sample to get validation set 
        if seed is not None:
            random.seed(seed)
        
        if subject_col in train_df.columns:
            val_spks = train_df[subject_col].drop_duplicates().to_list()
            val_spks = random.sample(val_spks, val_size)
            colid = subject_col
            
            train_df = train_df.loc[~train_df[subject_col].isin(val_spks)]
            val_df = train_df.loc[train_df[subject_col].isin(val_spks)]

            
        else:
            uids = train_df[uid_col].to_list()
            random.shuffle(uids)
            val_spks = uids[:val_size]
            colid = uid_col

        train_df = train_df.loc[~train_df[colid].isin(val_spks)]
        val_df = train_df.loc[train_df[colid].isin(val_spks)]
            
    #save validation set
        val_path = output_dir / 'validation.csv'

        if cloud['output']:
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_path = Path(tmpdirname) / 'validation.csv'
                val_df.to_csv(local_path, index=True)

                upload_to_gcs(val_path, local_path, bucket)
        else:
            val_df.to_csv(val_path, index=True)

    #alter data columns
    #remove NA

    try:
        train_df=train_df.dropna(subset=target_labels)
        val_df=val_df.dropna(subset=target_labels)
        test_df=test_df.dropna(subset=target_labels)
    except:
        raise ValueError('Target labels not included in train/val/test splits. Please include OR give a path to an annotation csv.')

    return train_df, val_df, test_df