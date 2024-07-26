import glob
import pandas as pd
import random
import tempfile
import json

from pathlib import Path
from typing import List, Union

from master_audio.io import search_gcs, upload_to_gcs


def generate_datasplit(input_dir: Union[str, Path], datasplit_dir: Union[str, Path], cloud: dict, 
                       train_proportion:float=.8, val_size:int=50, bucket=None, metadata_csv: Union[str,Path]=None, 
                       uid_col: str = 'originalaudioid', subject_col: str='record') -> tuple[Path, Path, Path]:
    """
    Generate a datasplit. Must also have a csv of labels in order to work.

    :param input_dir: path to input directory with data
    :param datasplit_dir: path to save datasplits to
    :param cloud: dictionary with booleans for which directories are stored in cloud vs. local
    :param train_proportion: proportion of data to include in training directory (between 0-1). Whether it is based on files or participants depends on whether a subject column is included in metadata
    :param val_size: number of either participants (if subject column in metadata) or files to include in validation set
    :param bucket: cloud bucket
    :param metadata_csv: path to metadata csv with labels and at minimum a uid column
    :param uid_col: column storing the uids
    :param subject_col: column storing the subject_col if included in metadata
    :return: paths to the datasplit csvs
    """
    input_dir = Path(input_dir)
    datasplit_dir = Path(datasplit_dir)

    if any(cloud.values()):
        assert bucket is not None, 'Must give bucket if using cloud.'

    if cloud['input']:
        files = search_gcs('waveform.wav', input_dir, bucket)
          
    else:
        files = glob.glob(str(input_dir / '*/waveform.wav'))
    

    md = pd.read_csv(str(metadata_csv))

    uids = list(set([str(Path(f).parents[0].name) for f in files]))

    
    if subject_col in md.columns:
        #md = md[[uid_col, subject_col]]
        md = md.rename(columns={uid_col: 'uid', subject_col: 'subject'})
        colid = 'subject'
    else:
        #md = md[[uid_col]]
        md = md.rename(columns={uid_col: 'uid'})
        colid='uid'

    md = md.dropna(subset=colid)   
        #md = md.set_index('uid')
    uid_df = pd.DataFrame({'uid':uids})
    #uid_df = uid_df.set_index('uid')
    uid_df = pd.merge(left=uid_df, left_on='uid', right=md, right_on='uid', how="inner")

    subjects = list(set(uid_df[colid].to_list()))
    random.shuffle(subjects)

    train_size = int(len(subjects)*train_proportion)
    train = subjects[:train_size]
    val = train[:val_size]
    train = train[val_size:]
    test = subjects[train_size:]

    train_df = pd.DataFrame({colid: train})
    #train_df = train_df.set_index('uid')
    val_df = pd.DataFrame({colid:val})
    #val_df = val_df.set_index('uid')
    test_df = pd.DataFrame({colid:test})
    

    train_df = pd.merge(left=uid_df, left_on=colid, right=train_df, right_on=colid, how="inner")
    test_df = pd.merge(left=uid_df, left_on=colid, right=test_df, right_on=colid, how="inner")
    val_df = pd.merge(left=uid_df, left_on=colid, right=val_df, right_on=colid, how="inner")
    
    train_df = train_df.set_index('uid')
    val_df = val_df.set_index('uid')
    test_df = test_df.set_index('uid')

    output_train = datasplit_dir / 'train.csv'
    output_test = datasplit_dir / 'test.csv'
    output_val = datasplit_dir /'validation.csv'

    if cloud['datasplit']:
        with tempfile.TemporaryDirectory() as tmpdirname:
            train_path = Path(tmpdirname) / 'train.csv'
            val_path = Path(tmpdirname) /'validation.csv'
            test_path = Path(tmpdirname) / 'test.csv'
            train_df.to_csv(train_path, index=True)
            val_df.to_csv(val_path, index=True)
            test_df.to_csv(test_path, index=True)
            upload_to_gcs(output_train, train_path, bucket)
            upload_to_gcs(output_test, test_path, bucket)
    else:
        train_df.to_csv(output_train, index=True)
        test_df.to_csv(output_test, index=True)
        val_df.to_csv(output_val, index=True)
    
    return output_train, output_val, output_test
