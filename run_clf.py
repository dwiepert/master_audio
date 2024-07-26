"""
"""
import argparse
import ast
import glob
import itertools
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
from google.cloud import storage
import torch

from master_audio.io import download_checkpoint_from_gcs, upload_to_gcs, search_gcs, download_file_to_local
from master_audio.dataset import generate_datasplit
from master_audio.models.classification import *
from master_audio.tasks import ClassificationWrapper

def _check_args(args):
    """
    """

    if args.model_type == 'w2v2':
        #check the following params
        
        assert args.mode in ['finetune','evaluate','extract'], f'Given mode {args.mode} is not available for W2V2 models.'
        if args.model_size == 'base':
            assert args.layer >= -1 and args.layer <= 12, f'Layer {args.layer} is not a valid choice for base W2V2 models.'
        else:
            raise NotImplementedError(f'{args.model_size} is not an implemented size for W2V2 models.')
        
        assert args.pooling_mode in ['mean','sum','max'], f'{args.pooling_mode} is not a valid pooling mode.'
        

    elif args.model_type == 'ssast':
       
        assert args.mode in ['pretrain','finetune','evaluate','extract'], f'Given mode {args.mode} is not available for W2V2 models.'
        assert args.ssast_task in ["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"], f'{args.ssast_task} is not a valid task for SSAST model.'
        if args.model_size == 'base':
            assert args.layer >= -1 and args.layer <= 12, f'Layer {args.layer} is not a valid choice for base SSAST models.'
        else:
            raise NotImplementedError(f'{args.model_size} is not an implemented size for SSAST models.')
        
        #TODO: check on pooling mode! Can this one take max?
        if args.mode == 'pretrain':
            assert args.ssast_task in ['pretrain_mpc', 'pretrain_mpg', 'pretrain_joint']
        else:
            assert args.ssast_task in ["ft_avgtok", "ft_cls"] 

    else:
        raise NotImplementedError(f'{args.model_type} is not an implemented classification model.')

    # (3) Check values
    if args.data_split_root is None:
        assert args.metadata_csv is not None
        if args.mode in ['evaulate', 'extract']:
            args.data_split_root = args.metadata_csv
        assert args.train_proportion is not None, 'Must specify size of train/test split if not giving a path to a directory containing a train/test split'
    
    if args.mode in ['evaluate', 'extract']:
        assert '.csv' in args.data_split_root[-4:], 'Data split root must be a path to a csv if evaluating or extracting.'
    elif args.data_split_root is not None:
        assert '.csv' not in args.data_split_root[-4:], 'Data split root must be a dir if finetuning.'

    if args.mode == 'extract':
        assert args.embedding_type in ['ft','pt', 'wt', 'st', None], f'{args.embedding_type} is not a valid embedding type.'
    
    if args.clip_length == 0:
        args.batch_size = 1 # 'Not currently compatible with different length wav files unless batch size has been set to 1'

    if args.metadata_csv is not None:
        assert '.csv' in args.metadata_csv[-4:]

    assert args.label_txt is not None
    
    return args

def _check_cloud(args):
    # (2) Data input/output/cloud argument checks
    if any(list(args.cloud.values())):
        assert args.project_name is not None, 'Must give a project name for use with cloud.'
        assert args.bucket_name is not None, 'Must give a bucket name for use with cloud.'
        client = storage.Client(args.project_name)
        args.bucket = client.get_bucket(args.bucket_name)
    else:
        args.bucket = None

    if args.cloud['checkpoint']:
        local_checkpoint = args.local_dir / f'checkpoints/{os.path.basename(args.checkpoint)}' 
        download_checkpoint_from_gcs(checkpoint_prefix = args.checkpoint, local_path = local_checkpoint, bucket=args.bucket)
        args.checkpoint = local_checkpoint

    if args.cloud['finetuned_mdl'] and args.finetuned_mdl_path is not None:
        local_checkpoint = args.local_dir / f'finetuned_mdls/{os.path.basename(args.finetuned_mdl_path)}' 
        print('TODO: issue may appear if trying to load .pt file')
        download_checkpoint_from_gcs(checkpoint_prefix=args.finetuned_mdl_path, local_path = local_checkpoint, bucket=args.bucket)
        args.finetuned_mdl_path = local_checkpoint

    if args.cloud['datasplit']:
        assert 'gs://' in args.metadata_csv, 'Must give full gs:// path for metadata csv'
    # else:
    #     local_datasplit = args.local_dir / 'datasplits'
    #     download_checkpoint_from_gcs(checkpoint_prefix=args.data_split_root, local_path = local_datasplit, bucket=args.bucket)
    #     args.data_split_root = local_datasplit

    return args

def _load_targets(args):
    if args.label_txt is None:
        assert args.mode == 'extract', 'Must give a txt with target labels for training or evaluating.'
        args.target_labels = None
        args.label_groups = None
        args.n_class = []
    else:
        if args.cloud['labels']:
            with tempfile.TemporaryDirectory() as tmpdirname:
                blob = args.bucket.blob(str(args.label_txt))
                local_path = Path(tmpdirname) / args.label_txt.name
                blob.download_to_filename(local_path)
                with open(local_path) as f:
                    target_labels = f.readlines()
        else:
            with open(args.label_txt) as f:
                target_labels = f.readlines()

        target_labels = [l.strip().split(sep=",") for l in target_labels]
        args.label_groups = target_labels 
        args.target_labels = list(itertools.chain.from_iterable(target_labels))
        args.n_class = [len(l) for l in args.label_groups]

        if args.n_class == []:
            assert args.mode == 'extract', 'Target labels must be given for training or evaluating. Txt file was empty.'
    return args 

def _create_dirs(args):
    args.input_dir = Path(args.input_dir)
    if args.label_txt is not None:
        args.label_txt = Path(args.label_txt)
    args.output_dir = Path(args.output_dir)
    args.checkpoint = Path(args.checkpoint)
    if args.data_split_root is not None and 'gs://' not in args.data_split_root:
        args.data_split_root = Path(args.data_split_root)
    if args.finetuned_mdl_path is not None:
        args.finetuned_mdl_path = Path(args.finetuned_mdl_path)
    args.local_dir = Path(args.local_dir).absolute()

    if not args.cloud['input']:
        assert args.input_dir.exists(), f'Input directory {args.input_dir} does not exist locally.'
    
    if not args.cloud['output']:
        if not args.output_dir.exists():
            os.makedirs(args.output_dir)

    if any(list(args.cloud.values())) and not args.local_dir.exists():
        os.makedirs(args.local_dir)

    if not args.cloud['labels'] and args.label_txt is not None:
        args.label_txt = args.label_txt.absolute()        
        assert args.label_txt.exists(), f'Label txt file {args.label_txt} does not exist locally.'

    if not args.cloud['checkpoint']:
        args.checkpoint = args.checkpoint.absolute()
        assert args.checkpoint.exists(), f'Checkpoint {args.checkpoint} does not exist locally.'

    if not args.cloud['finetuned_mdl'] and args.finetuned_mdl_path is not None:
        args.finetuned_mdl_path = args.finetuned_mdl_path.absolute()
        assert args.finetuned_mdl_path.exists(), f'Finetuned model path {args.finetuned_mdl_path} does not exist locally.'
    
    if not args.cloud['datasplit'] and args.data_split_root is not None:
        args.data_split_root = args.data_split_root.absolute()
        assert args.data_split_root.exists(), f'{args.data_split_root} does not exist locally.'
    return args

def _dump_arguments(args):
    if args.mode=='finetune':
        args_temp = args
        args_temp.bucket = None
        #only save args if training a model. 

        if args.cloud['output']:
            with tempfile.TemporaryDirectory() as tmpdirname:
                args_path = Path(tmpdirname) / "args.pkl" 
                with open(args_path, "wb") as f:
                    pickle.dump(args_temp, f)
                
                upload_to_gcs(args.output_dir, args_path, args.bucket)
        else:
            args_path =  args.output_dir / "args.pkl" 
            try:
                assert not os.path.exists(args_path)
            except:
                print('Current experiment directory already has an args.pkl file. This file will be overwritten. Please change experiment directory or rename the args.pkl to avoid overwriting the file in the future.')

            with open(args_path, "wb") as f:
                pickle.dump(args_temp, f)

def _generate_name(args):
    if args.mode == 'finetune' and args.hp_tuning:
        path =  '{}_{}_{}{}_epoch{}_{}_clf{}_lr{}_bs{}_layer{}'.format(args.dataset, np.sum(args.n_class), args.optim,args.weight_decay, args.epochs, os.path.basename(args.checkpoint), len(args.n_class), args.learning_rate, args.batch_size, args.layer)
        if args.shared_dense:
            path = path + '_sd{}'.format(args.sd_bottleneck)
        if args.weighted:
            path = path + '_ws'
        if args.scheduler is not None:
            path = path + '_{}_maxlr{}'.format(args.scheduler, args.max_lr)
        if args.layernorm:
            path = path + '_layernorm'
        path = path + '_dropout{}_clf{}'.format(args.final_dropout, args.clf_bottleneck)
        args.output_dir = args.output_dir /path
    
    return args

def _check_datasplit(args):
    if args.mode not in ['extract', 'evaluate']:
        if args.data_split_root is None:
            generate = True
        else:
            train_csv = args.data_split_root / 'train.csv'
            test_csv = args.data_split_root / 'test.csv'
            if not train_csv.exists() or not test_csv.exists():
                generate = True
            else:
                generate = False

        if generate:
            args.data_split_root = args.output_dir
            if args.cloud['output']:
                args.cloud['datasplit'] = True 
            else:
                args.cloud['datasplit'] = False
            _ = generate_datasplit(args.input_dir, args.data_split_root, args.cloud, args.train_proportion, args.val_size, args.bucket, args.metadata_csv, args.md_uid_col, args.md_subject_col)

    # else:
    #     if args.data_split_root is None:
    #         args.data_split_root =     args.output_dir
    #         _ = generate_datasplit(args.input_dir, args.data_split_root, args.cloud, 0, 0, args.bucket, args.metadata_csv, args.md_uid_col, args.md_subject_col) #assumes that all values should be in the test.csv
            
    return args 

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('--input_dir', default = '', help='Set path to directory with wav files to process. Can be local directory or google cloud storage bucket.')
    parser.add_argument("-d", "--data_split_root", default=None, help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.  If cloud and .csv, use full gs:// path.")
    parser.add_argument("--metadata_csv", default='', help="specify if there is a csv matching uids to subjects and including annotation data. If cloud, use full gs:// path. ")
    parser.add_argument("--md_uid_col", default="originalaudioid")
    parser.add_argument("--md_subject_col", default="record")
    parser.add_argument("--train_proportion", default=.8, help="specify size of train/test split")
    parser.add_argument('-l','--label_txt', default=None) #default=None #default='./labels.txt'
    #Outputs
    parser.add_argument("--save_logs", default=True, type=ast.literal_eval, help="Specify whether to save outputs.")
    parser.add_argument("--output_dir", default= '', help='Set path to directory where outputs should be saved.')
    #GCS
    parser.add_argument("--cloud",  nargs="+", type=ast.literal_eval, default=[False, False, False, False, False, False], help="Specify which files are located on the cloud/should be located on the cloud [input_dir, label_txt, output_dir, checkpoint, finetuned_mdl_path, data_split_root]")
    parser.add_argument("--local_dir", default='', help="Specify location to save files downloaded from bucket")
    parser.add_argument('-b','--bucket_name', default='', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='', help='google cloud platform project name')
    #Mode specifics
    parser.add_argument("-m", "--mode", choices=['pretrain','finetune','evaluate','extract'], default='finetune')
    parser.add_argument("--weighted", type=ast.literal_eval, default=False, help="specify whether to learn a weighted sum of layers for classification")
    parser.add_argument("--layer", default=-1, type=int, help="specify which hidden state is being used. It can be between -1 and 12")
    parser.add_argument("--freeze", type=ast.literal_eval, default=True, help='specify whether to freeze the base model')
    parser.add_argument("--shared_dense", type=ast.literal_eval, default=False, help="specify whether to add an additional shared dense layer before the classifier(s)")
    parser.add_argument("--sd_bottleneck", type=int, default=768, help="specify whether to decrease when using shared_dense layer")
    parser.add_argument('--embedding_type', type=str, default='ft', help='specify whether embeddings should be extracted from classification head (ft), base pretrained model (pt), weighted sum (wt),or shared dense layer (st)', choices=['ft','pt', 'wt', 'st', None])
    parser.add_argument("--ssast_task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"]) #SSAST specifics
    #Model specifics
    parser.add_argument("--model_type", default="ssast", choices=["w2v2","ssast"], help="specify model type")
    parser.add_argument('--model_size', default='base',help='the size of the model', type=str)
    parser.add_argument("-c", "--checkpoint", default='', help="specify path to pre-trained model weight checkpoint")
    parser.add_argument("-mp", "--finetuned_mdl_path", default=None, help='If running eval-only or extraction, you have the option to load a fine-tuned model by specifying the model path')
    parser.add_argument("--seed", default=4200, help='Specify a seed for random number generator to make validation set consistent across runs. Accepts None or any valid RandomState input (i.e., int)')
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max']) #make sure that model specific values are accounted for (ssast only takes mean/sum?)
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu',choices=["relu"], help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.3, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=ast.literal_eval, default=False, help="specify whether to include the LayerNorm in classification head")
    parser.add_argument("--clf_bottleneck", type=int, default=768, help="specify whether to apply a bottleneck to initial classifier dense layer")
    #SSAST specific model parameters
    parser.add_argument("--fstride", type=int, default=128,help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, default=2, help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument("--fshape", type=int, default=128,help="shape of patch on the frequency dimension")
    parser.add_argument("--tshape", type=int, default=2, help="shape of patch on the time dimension")
    parser.add_argument("--target_length", default=1024, type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", default=128,type=int, help="number of input mel bins")
    #SSAST specific Pretraining parameters
    parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
    parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
    #Dataset specifics/transforms
    parser.add_argument("--val_size", default=50, type=int, help="Specify size of validation set to generate")
    parser.add_argument('--use_librosa', default=False, type=ast.literal_eval, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--monochannel", default=True, type=ast.literal_eval, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=10.0, type=float, help="If truncating audio, specify clip length in seconds. 0 = no truncation")
    parser.add_argument("--trim_level", default=7, type=float, help="specify db threshold for trimming beginning/end silence (60?) if librosa or trigger level threshold for torchaudio (7?).")
    parser.add_argument("--padding", default='do_not_pad', type=str, help="longest/do_not_pad/max_length")
   
    parser.add_argument("--gauss", default=None, type=float, help="Specify p for AddGaussianNoise")
    parser.add_argument("--gausssnr", default=None, type=float, help="Specify p for AddGaussianNoiseSNR")
    parser.add_argument("--alias", default=None, type=float, help="Specify p for Aliasing")
    parser.add_argument("--bandstop", default=None, type=float, help="Specify p for BandStopFilter")
    parser.add_argument("--bitcrush", default=None, type=float, help="Specify p for BitCrush")
    parser.add_argument("--clipd", default=None, type=float, help="Specify p for ClippingDistortion")
    parser.add_argument("--gain", default=None, type=float, help="Specify p for Gain")
    parser.add_argument("--gaint", default=None, type=float, help="Specify p for GainTransition")
    parser.add_argument("--mp3", default=None, type=float, help="Specify p for Mp3Compression")
    parser.add_argument("--norm", default=None, type=float, help="Specify p for Normalize")
    parser.add_argument("--pshift", default=None, type=float, help="Specify p for PitchShift")
    parser.add_argument("--pinversion", default=None, type=float, help="Specify p for PolarityInversion")
    parser.add_argument("--tstretch", default=None, type=float, help="Specify p for TimeStretch")
    parser.add_argument("--tmask", default=None, type=float, help="Specify p for TimeMask")
    parser.add_argument("--tanh", default=None, type=float, help="Specify p for TanhDistortion")
    parser.add_argument("--repeat", default=None, type=float, help="Specify p for RepeatPart")
    parser.add_argument("--reverse", default=None, type=float, help="Specify p for Reverse")
    parser.add_argument("--room", default=None, type=float, help="Specify p for RoomSimulator")
    parser.add_argument("--tshift", default=None, type=float, help="Specify p for Shift")
    parser.add_argument("--mixup", default=None)
    #SSAST/Spectrogram specifics
    parser.add_argument("--dataset_mean", default=-4.2677393, type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", default=4.5689974, type=float, help="the dataset std, used for input normalization")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument('--timem_p', help='specify probability for time masking', type=float, default=0)
    parser.add_argument("--noise", type=ast.literal_eval, default=False, help="specify if augment noise in finetuning")
    #Training Parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adamw", help="training optimizer", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=.0001, help='specify weight decay for adamw')
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", "None",None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #OTHER
    parser.add_argument("--debug", default=False, type=ast.literal_eval)
    parser.add_argument("--hp_tuning", default=False, type=ast.literal_eval)
    parser.add_argument("--new_checkpoint", default=False, type=ast.literal_eval, help="specify if you should use the checkpoint specified in current args for eval")
    args = parser.parse_args()

    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print('Checking arguments...')
    args = _check_args(args)

    assert len(args.cloud) == 6, 'Must have a True/False value for all directory/file inputs'
    args.cloud = {'input':args.cloud[0], 'labels': args.cloud[1],'output':args.cloud[2], 'checkpoint': args.cloud[3], 'finetuned_mdl': args.cloud[4], 'datasplit': args.cloud[5]}
    
    args = _create_dirs(args)

    args = _check_cloud(args)
    args = _load_targets(args)
    args = _check_datasplit(args)

    _dump_arguments(args)
    #set up config dicts

    if args.mode in ['evaluate', 'extract'] and args.finetuned_mdl_path is not None:
        print('TODO: LOAD PREVIOUS ARGS')
        search_dir = args.finetuned_mdl_path.parents[0]
        if args.cloud['finetuned_mdl']:
            file = search_gcs('args.pkl', search_dir, args.bucket)
            if file != []:
                with tempfile.TemporaryDirectory() as tempdir:
                    new_path = download_file_to_local(file[0], Path(tempdir) / 'args.pkl', args.bucket)
                    with open(new_path, 'rb') as f:
                        model_args = pickle.load(f)
            else:
                print('No args.pkl file present with loaded model. Assuming current arguments.')
                model_args = args
        else:
            file = glob.glob(str(search_dir / 'args.pkl'))
        
            if file != []:
                with open(file[0], 'rb') as f:
                    model_args = pickle.load(f)
            else:
                print('No args.pkl file present with loaded model. Assuming current arguments.')
                model_args = args   

        model_args.checkpoint = Path(model_args.checkpoint).absolute()
        if not model_args.checkpoint.exists:
            if model_args.cloud['checkpoint']:
                local_checkpoint = args.local_dir / f'checkpoints/{os.path.basename(model_args.checkpoint)}' 
                download_checkpoint_from_gcs(checkpoint_prefix = model_args.checkpoint, local_path = local_checkpoint, bucket=args.bucket)
                model_args.checkpoint = local_checkpoint
            else:
                raise ValueError(f'Check that the checkpoint {model_args.checkpoint} exists locally.')
        

        #NEED ARGS CHECKPOINT? ARGS FINETUNED MDL PATH?
        model_config = {'checkpoint':model_args.checkpoint,'ssast_task': args.ssast_task, 'mode': args.mode, 'finetuned_mdl_path':args.finetuned_mdl_path, 'embedding_type': args.embedding_type,
                        'seed':model_args.seed, 'pooling_mode':model_args.pooling_mode,
                        'freeze':model_args.freeze, 'weighted':model_args.weighted, 'layer':model_args.layer, 'shared_dense':model_args.shared_dense, 
                        'sd_bottleneck': model_args.sd_bottleneck, 'clf_bottleneck': model_args.clf_bottleneck, 'activation': model_args.activation, 
                        'final_dropout': model_args.final_dropout, 'layernorm': model_args.layernorm, 'n_class': model_args.n_class, 'label_groups': model_args.label_groups,
                        'fstride': model_args.fstride, 'tstride':model_args.tstride, 'fshape':model_args.fshape, 'tshape':model_args.tshape,
                        'target_length': model_args.target_length, 'num_mel_bins': model_args.num_mel_bins,
                        'mask_patch': model_args.mask_patch, 'cluster_factor':model_args.cluster_factor}

        args.model_type = model_args.model_type
        args.model_size = model_args.model_size

        data_config = {'val_size':args.val_size, 'seed': model_args.seed, 'target_labels': model_args.target_labels, 'use_librosa':model_args.use_librosa,
                        'resample_rate':model_args.resample_rate, 'monochannel':model_args.monochannel, 'clip_length': args.clip_length,
                    'trim_level': args.trim_level, 'dataset_mean':args.dataset_mean, 'dataset_std':args.dataset_std, 'padding':'do_not_pad'}

    elif args.mode in ['evaluate'] and args.finetuned_mdl_path is None:
        raise NotImplementedError()
    elif args.mode in ['extract'] and args.embedding_type != 'pt' and args.finetuned_mdl_path is None:
        raise NotImplementedError()
    else:
        model_config = {'checkpoint': args.checkpoint, 'mode': args.mode, 'seed': args.seed, 'pooling_mode': args.pooling_mode,
                        'freeze': args.freeze, 'weighted': args.weighted, 'layer':args.layer, 'shared_dense': args.shared_dense, 
                        'sd_bottleneck': args.sd_bottleneck, 'clf_bottleneck': args.clf_bottleneck, 'activation': args.activation, 
                        'final_dropout': args.final_dropout, 'layernorm': args.layernorm, 'n_class': args.n_class, 'label_groups': args.label_groups,
                        'finetuned_mdl_path':args.finetuned_mdl_path, 'embedding_type': args.embedding_type,
                        'fstride': args.fstride, 'tstride':args.tstride, 'fshape': args.fshape, 'tshape': args.tshape,
                        'ssast_task': args.ssast_task, 'target_length': args.target_length, 'num_mel_bins': args.num_mel_bins,
                        'mask_patch': args.mask_patch, 'cluster_factor':args.cluster_factor}
    
        data_config = {'val_size':args.val_size, 'seed': args.seed, 'target_labels': args.target_labels, 'use_librosa':args.use_librosa,
                    'resample_rate':args.resample_rate, 'monochannel':args.monochannel, 'clip_length': args.clip_length,
                        'trim_level': args.trim_level, 'padding':args.padding, 'gauss': args.gauss, 'gausssnr':args.gausssnr, 'alias':args.alias, 
                        'bandstop':args.bandstop, 'bitcrush': args.bitcrush, 'clipd':args.clipd, 'gain':args.gain,
                        'gaint':args.gaint, 'mp3': args.mp3, 'norm': args.norm, 'pshift':args.pshift, 'pinversion':args.pinversion,
                        'tstretch': args.tstretch, 'tmask':args.tmask, 'tanh':args.tanh, 'repeat':args.repeat, 'reverse':args.reverse,
                        'room': args.room, 'tshift':args.tshift, 'mixup':args.mixup, 'dataset_mean':args.dataset_mean, 'dataset_std':args.dataset_std,
                        'freqm':args.freqm, 'timem':args.timem, 'timem_p':args.timem_p, 'noise':args.noise}
    
    train_config = {'batch_size':args.batch_size, 'num_workers':args.num_workers, 'epochs':args.epochs,
                    'optim':args.optim, 'weight_decay':args.weight_decay, 'learning_rate':args.learning_rate,
                    'loss':args.loss, 'scheduler':args.scheduler, 'max_lr':args.max_lr}
   
    if args.model_type == 'w2v2' and args.mode in ['extract', 'evaluate'] and not args.padding:
        train_config['batch_size'] = 1
    
    wrap = ClassificationWrapper(input_dir=args.input_dir, data_split_root=args.data_split_root, uid_col=args.md_uid_col, 
                          output_dir = args.output_dir, model_type=args.model_type, model_size=args.model_size, model_config=model_config, 
                          data_config=data_config, train_config=train_config, cloud=args.cloud, bucket=args.bucket, debug=args.debug)

    wrap()

if __name__ == "__main__":
    main()