import argparse
import ast
import os
from pathlib import Path
import json
import glob
import shutil
import tempfile
from typing import Union
from google.cloud import storage
import numpy as np
import pandas as pd

from master_audio.dataset import WaveDataset, collate_asr
from master_audio.models.asr import W2V2ForASR, WhisperForASR
from master_audio.io import upload_to_gcs, search_gcs, download_checkpoint_from_gcs
from torch.utils.data import  DataLoader
from tqdm import tqdm


def save_results(results, output_dir:Path, model_name, bucket = None):
    """
    """
    json_string = json.dumps(results, indent=4)
    to_add = model_name+'_ASR_results.json'
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

def update_metadata(metadata, prefix:Union[str,Path], results:dict, transcription_name, cloud, bucket):
    path = Path(prefix) / 'metadata.json'


    metadata[transcription_name] = results

    if cloud:
        with tempfile.TemporaryDirectory() as tempdir:
            #temppath = Path(tempdir) /'metadata1.json'
            temppath = Path(tempdir) /'metadata.json'
            json_string = json.dumps(metadata, indent=4)
            with open(temppath, 'w') as outfile:
                outfile.write(json_string)

            
            upload_to_gcs(gcs_prefix=path, path=temppath, bucket=bucket)
    else:
        json_string = json.dumps(metadata, indent=4)
        with open(path, 'w') as outfile:
            outfile.write(json_string)

def run(args):
    """
    """
    if any(list(args.cloud.values())):
        data = search_gcs('waveform.wav', args.input_dir, args.bucket)
    else:
        pat = args.input_dir / '*/waveform.wav'
        data = glob.glob(str(pat))#get full file paths in the input directory
    
    if args.model_type == 'w2v2':
        model = W2V2ForASR(args.checkpoint, args.model_size)
    elif args.model_type == 'whisper':
        model = WhisperForASR(args.checkpoint, args.model_size)
    else:
        raise NotImplementedError(f'{args.model_type} is not an implemented ASR model.')

    results = {}
    for d in data:
        r = model.transcribe(d, return_timestamps=args.return_timestamps, return_pauses=args.return_pauses, pause_s=args.pause_s)
        r['pause_s'] = args.pause_s
        results[d] = r
    return results

def run_with_dataset(args):
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
    ASRdataset = WaveDataset(data = data, prefix=args.input_dir,  model_type=args.model_type,
                             model_task='asr', dataset_config=dataset_config, target_labels=None, bucket=args.bucket)

    #Batchsize should ALWAYS BE 1 For ASR dataset
    ASRdataloader = DataLoader(ASRdataset, collate_fn=collate_asr, batch_size = 1)

    #(2) ASR Model set up
    if args.model_type == 'w2v2':
        model = W2V2ForASR(args.checkpoint, args.model_size)
    elif args.model_type == 'whisper':
        model = WhisperForASR(args.checkpoint, args.model_size)
    else:
        raise NotImplementedError(f'{args.model_type} is not an implemented ASR model.')

    results = {}

    for batch in tqdm(ASRdataloader):
        transcription_name = args.model_type+'_'+args.model_size+'_transcription'
        metadata = batch['metadata'][0]
        uid = batch['uid'][0]
        audio = batch['waveform'][0] #NOTE THAT WITH THE DATASET, THE AUDIO DATA, WHETHER LOADED OR A PATH, IS ALWAYS UNDER 'waveform'

        if metadata['task'] not in args.tasks:
            if args.load_waveform is False and args.bucket is not None:
                shutil.rmtree(os.path.dirname(audio))
            continue 

        if transcription_name in metadata and not args.override:
            results[uid] = metadata[transcription_name]
            if args.load_waveform is False and args.bucket is not None:
                shutil.rmtree(os.path.dirname(audio))
            continue
        
        if 'sample_rate' in batch:
            sample_rate = int(batch['sample_rate'][0])
        else:
            sample_rate = None
        print(uid)
        r = model.transcribe(audio, samplerate = sample_rate, return_timestamps=args.return_timestamps, return_pauses=args.return_pauses, pause_s=args.pause_s)
        r['pause_s'] = args.pause_s
        results[uid] = r
        update_metadata(metadata = metadata, prefix = args.input_dir / uid, results = r, transcription_name = transcription_name, cloud = args.cloud['input'], bucket=args.bucket)

        if args.load_waveform is False and args.bucket is not None:
            shutil.rmtree(os.path.dirname(audio))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '', help='Set path to directory with wav files to process. Can be local directory or google cloud storage bucket.')
    parser.add_argument("--save_outputs", default=True, type=ast.literal_eval, help="Specify whether to save outputs.")
    parser.add_argument("--output_dir", default= '', help='Set path to directory where outputs should be saved.')
    parser.add_argument("--override", default=False, type=ast.literal_eval, help='specify whether to rerun and overwrite currently existing transcription')
    parser.add_argument("--tasks", nargs="+", default=['Sentence Repetition', 'Word Repetition: Catapult', 'Word Repetition: Catastrophe', 'Animal Fluency', 'Counting', 'Picture Description', 'Reading', 'Word Repetition: Catastrophe Slow', 'Word Repetition: Catastrophe Fast'], help = 'specify which tasks to get transcriptions for.')
    #run methods
    parser.add_argument("--rundataset",default=True, help="Specify whether to run using torch Dataset. If false, defaults to regular for loop/lists and only uses audio paths (does not load outside of model)." )
    #model specifics
    parser.add_argument("--model_type", default='whisper', choices = ['w2v2','whisper'], help='Specify model to use.')
    parser.add_argument("--model_size", default="medium", help='Specify model size.')
    parser.add_argument("--checkpoint", default = 'medium.pt', help='Specify model checkpoint')
    #transcription specifics
    parser.add_argument("--return_timestamps", default=True, type=ast.literal_eval, help= "Specify whether to get timestamps for each audio file")
    parser.add_argument("--return_pauses", default=True, type=ast.literal_eval, help="Specify whether to find long pauses.")
    parser.add_argument("--pause_s", default=0.1, type=float, help='Set threshold for a long pause in SECONDS.')
    #audio loading specifics
    parser.add_argument('--load_waveform', default=False, type=ast.literal_eval, help='Specify whether to fully load waveform or to load when running the model.')
    parser.add_argument("--resample_rate", default=16000, type=int, help="Choose resample rate. Not required for ASR as it can be done within the models.")
    parser.add_argument("--monochannel", default=True, type=ast.literal_eval, help="Choose to reduce channels. Not required for ASR.")
    parser.add_argument("--clip_length", default=None, type=int, help="Choose whether to truncate audio. Not required for ASR.")
    parser.add_argument("--trim_db", default=60, type=float, help="specify db threshold for trimming beginning/end silence.")
    #GCS
    parser.add_argument("-c", "--cloud",  nargs="+", type=ast.literal_eval, default=[False, False, False], help="Specify which files are located on the cloud/should be located on the cloud [input_dir, output_dir, checkpoint]")
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
    

    if args.rundataset:
        results = run_with_dataset(args)
    else:
        results = run(args)

    if args.save_outputs:
        save_results(results, args.output_dir,model_name = args.model_type+'_'+args.model_size, bucket=args.bucket)

if __name__ == "__main__":
    main()
   