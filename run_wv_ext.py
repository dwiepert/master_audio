import argparse
import ast
import os
from pathlib import Path
import json
import glob
import tempfile
from typing import Union
from google.cloud import storage
import numpy as np
import pandas as pd

from master_audio.dataset import WaveDataset, collate_asr
from master_audio.models.word_vectors import Word2Vec_Extract, WordNet, FastText
from master_audio.io import upload_to_gcs, search_gcs, download_checkpoint_from_gcs
from torch.utils.data import  DataLoader
from tqdm import tqdm

def save_results(results, output_dir:Path, model_name, bucket = None):
    """
    """
    json_string = json.dumps(results, indent=4)
    to_add = model_name+'_wordsim_results.json'
    outpath = output_dir / to_add
    if bucket is None:
        with open(outpath, 'w') as outfile:
            outfile.write(json_string)
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppath = Path(tmpdirname) / to_add
            with open(tmppath, 'w') as outfile:
                outfile.write(json_string)
            
            upload_to_gcs(outpath, tmppath, bucket)

def run(args):
    """
    """
   #(1) Make dataset
    dataset_config = {'use_librosa': True, 'load_waveform': args.load_waveform, 'resample_rate': args.resample_rate, 'monochannel': args.monochannel, 'clip_length':args.clip_length,
                      'trim_db':args.trim_db}
    if any(list(args.cloud.values())):
        data = search_gcs('waveform.wav', args.input_dir, args.bucket)
        dataset_config['savedir'] = args.local_dir
        #list blobs
    else:
        pat = args.input_dir / '*/waveform.wav'
        data = glob.glob(str(pat))
    data = np.asarray([Path(d).parents[0].name for d in data])
    # labels = np.zeros(data.shape)
    # data = pd.DataFrame({'uid':data, 'label':labels})
    # data = data.set_index('uid')
    #data = [] #get np array of uids in the input directory
    #also try with dataframe
    dataset = WaveDataset(data = data, prefix=args.input_dir,  model_type=args.model_type,
                             model_task='asr', dataset_config=dataset_config, target_labels=None, bucket=args.bucket)

    #Batchsize should ALWAYS BE 1 For ASR dataset
    dataloader = DataLoader(dataset, collate_fn=collate_asr, batch_size = 1)

    #(2) ASR Model set up
    if args.model_type == 'fasttext':
        model = FastText(args.checkpoint, args.model_type)
    elif args.model_type == 'wordnet':
        model = WordNet()
    elif args.model_type == 'word2vec':
        model = Word2Vec_Extract(args.checkpoint)
    else:
        raise NotImplementedError(f'{args.model_type} is not an implemented ASR model.')

   
    results = {}
    for batch in tqdm(dataloader):
        metadata = batch['metadata'][0]
        if metadata['task'] in args.tasks:
            continue
        uid = batch['uid'][0]
        transcription = metadata[args.transcription_type]['transcription']
        
        sim_matrix = model.get_similarity_matrix(transcription).tolist()
        
        results[uid] = {'transcription': transcription, 'similarity': sim_matrix}

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '', help='Set path to directory with wav files to process. Can be local directory or google cloud storage bucket.')
    parser.add_argument("--save_outputs", default=True, type=ast.literal_eval, help="Specify whether to save outputs.")
    parser.add_argument("--output_dir", default= '', help='Set path to directory where outputs should be saved.')
    parser.add_argument("--tasks", nargs="+", default=['Animal Fluency'], help = 'specify which tasks to get transcriptions for.')
    #model specifics
    parser.add_argument("--model_type", default='whisper', choices = ['fasttext', 'wordnet', 'word2vec'], help='Specify model to use.')
    parser.add_argument("--checkpoint", default = 'medium.pt', help='Specify model checkpoint')
    parser.add_argument("--transcription_type", default='w2v2_base', help='Specify what transcription to load from metadata.')
    #GCS
    parser.add_argument("--cloud",  nargs="+", type=ast.literal_eval, default=[False,False, False], help="Specify which files are located on the cloud/should be located on the cloud [input_dir, output_dir, checkpoint]")
    parser.add_argument("--local_dir", default='', help="Specify location to save files downloaded from bucket")
    parser.add_argument('-b','--bucket_name', default='', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='', help='google cloud platform project name')
    
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.local_dir = Path(args.local_dir).absolute()
    args.checkpoint = Path(args.checkpoint)

    #check for cloud
    assert len(args.cloud) == 3, 'Must have a True/False value for all directory/file inputs'
    if any(args.cloud):
        assert args.project_name is not None, 'Must give a project name for use with cloud.'
        assert args.bucket_name is not None, 'Must give a bucket name for use with cloud.'
        client = storage.Client(args.project_name)
        args.bucket = client.get_bucket(args.bucket_name)
    else:
        args.bucket = None

    args.cloud = {'input':args.cloud[0], 'output':args.cloud[1], 'checkpoint': args.cloud[2]}

    if not args.cloud['input']:
        args.input_dir = args.input_dir.absolute()
        assert args.input_dir.exists(), f'Input directory {args.input_dir} does not exist locally.'
    if not args.cloud['output']:
        args.output_dir = args.output_dir.absolute()
        if not args.output_dir.exists():
            os.makedirs(args.output_dir)
    if not args.cloud['checkpoint']:
        args.checkpoint = args.checkpoint.absolute()
        assert args.checkpoint.exists(), f'Checkpoint {args.checkpoint} does not exist locally.'
    else:
        args.checkpoint = Path(args.checkpoint)
        local_checkpoint = args.local_dir / 'checkpoints'
        download_checkpoint_from_gcs(checkpoint_prefix = args.checkpoint, local_path = local_checkpoint, bucket=args.bucket)
        args.checkpoint = local_checkpoint

    results = run(args)

    if args.save_outputs:
        save_results(results, args.output_dir,model_name = args.model_type, bucket=args.bucket)