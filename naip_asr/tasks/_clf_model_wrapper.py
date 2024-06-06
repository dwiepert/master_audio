import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import  DataLoader

from naip_asr.io import load_input_data
from naip_asr.dataset import WaveDataset, collate_clf
from naip_asr.models.classification import W2V2FeatureExtractor, W2V2ForClassification, ASTModel_pretrain, ASTModel_finetune
from ._classify import Classify


class ClassificationWrapper():
    """
    Wrapper for running Classification

    :param input_dir: input directory path
    :param data_split_root: path to directory with datasplit csvs, OR path to a csv if just evaluating/extracting embeddings
    :param uid_col: column in the datasplit csvs with uids
    :param output_dir: output directory path
    :param model_type: str, model implemented
    :param model_size: str, size of the model (e.g., 'base')
    :param model_config: dictionary with all of the model parameters
    :param data_config: dictionary with all of the data parameters
    :param train_config: dictionary with all of the train parameters
    :param cloud: dict, dictionary with booleans indicating which directories are in gcs
    :param bucket: gcs bucket
    :param debug: boolean to switch to debug mode, default = False

    """
    def __init__(self, input_dir: Union[str, Path], data_split_root:Union[str, Path], uid_col: str, output_dir: Union[str, Path], 
                 model_type: str, model_size: str, 
                 model_config:dict, data_config:dict, train_config:dict, 
                 cloud:dict, bucket = None, debug:bool = False):
        

        self.input_dir = Path(input_dir)
        self.data_split_root = data_split_root
        if 'gs://' not in str(data_split_root):
            self.data_split_root = Path(self.data_split_root)
        self.uid_col = uid_col
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.model_size = model_size
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.cloud = cloud
        self.bucket = bucket
        self.debug = debug

        print('Parsing configs...')
        self._parse_model_config()
        self._parse_data_config()
        self._parse_train_config()

        print('Loading model...')
        self._load_model()

        print('Loading data...')
        self._load_data()
    
    def _parse_model_config(self):
        """
        Parse model configs
        """
        include = ['checkpoint','mode', 'pooling_mode', 'freeze', 'weighted', 'layer', 'shared_dense', 'clf_bottleneck', 'activation', 'final_dropout', 'layernorm', 'n_class', 'label_groups', 'finetuned_mdl_path', 'seed']
        for i in include:
            assert i in self.model_config, f'{i} not in model config but is required. May just need to be set to None'

        self.checkpoint = self.model_config.get('checkpoint')
        self.mode = self.model_config.get('mode')
        self.pooling_mode = self.model_config.get('pooling_mode')
        self.freeze = self.model_config.get('freeze')
        self.weighted = self.model_config.get('weighted')
        self.layer = self.model_config.get('layer')
        self.shared_dense = self.model_config.get('shared_dense')
        self.sd_bottleneck = self.model_config.get('sd_bottleneck')
        self.clf_bottleneck = self.model_config.get('clf_bottleneck')
        self.activation = self.model_config.get('activation')
        self.final_dropout = self.model_config.get('final_dropout')
        self.layernorm = self.model_config.get('layernorm')
        self.n_class = self.model_config.get('n_class')
        self.label_groups = self.data_config.get('label_groups')
        self.finetuned_mdl_path = self.model_config.get('finetuned_mdl_path')
        self.model_seed = self.model_config.get('seed')

        if self.mode == 'extract':
            # extract specific model configs
            self.embedding_type = self.model_config.get('embedding_type')

        if self.model_type == 'ssast':
            # model specific model configs
            self.data_type = 'fbank'
            self.fstride = self.model_config.get('fstride')
            self.tstride = self.model_config.get('tstride')
            self.fshape = self.model_config.get('fshape')
            self.tshape = self.model_config.get('tshape')
            self.ssast_task =  self.model_config.get('ssast_task')
            self.target_length = self.model_config.get('target_length')
            self.num_mel_bins = self.model_config.get('num_mel_bins')

            if self.mode == 'pretrain':
                self.mask_patch = self.model_config.get('mask_patch')
                self.cluster_factor = self.model_config.get('cluster_factor')
        else:
            self.data_type = 'waveform'

    def _parse_data_config(self):
        """
        Parse data config
        """

        #check must include values are in the data config
        include = ['val_size', 'seed', 'target_labels', 'padding']
        for i in include:
            assert i in self.data_config, f'{i} not in data config but is required.'

        self.val_size = self.data_config.get('val_size')
        self.data_seed = self.data_config.get('seed')
        self.target_labels = self.data_config.get('target_labels')
        self.padding = self.data_config.get('padding')

        del self.data_config['val_size']
        del self.data_config['target_labels']

        self.eval_data_config = self.data_config
        self.train_data_config = self.data_config

        # remove data augmentation from eval data config
        set_none = ['gauss', 'gausssnr', 'alias', 'bandstop', 'bitcrush', 'gain', 'gaint', 'mp3', 'norm', 'pshift', 'pinversion', 'tstretch', 'tmask', 'tanh', 'repeat', 'reverse', 'room', 'tshift', 'freqm', 'timem', 'noise']
        for s in set_none:
            if s in self.eval_data_config:
                del self.eval_data_config[s]
            
        if self.model_type == 'ssast':
            #model specific params
            if 'num_mel_bins' not in self.train_data_config:
                self.train_data_config['num_mel_bins'] = self.num_mel_bins
            if 'target_length' not in self.train_data_config:
                self.train_data_config['target_length'] = self.target_length
            
    def _parse_train_config(self):
        """
        Parse train config
        """
        include = ['batch_size', 'num_workers']
        for i in include:
            assert i in self.train_config, f'{i} not in train config but is required.'
        self.batch_size = self.train_config.get('batch_size')
        self.num_workers = self.train_config.get('num_workers')

        if self.mode in ['finetune', 'pretrain']:
            include = ['optim', 'learning_rate', 'weight_decay', 'loss', 'scheduler', 'max_lr', 'epochs']
            for i in include:
                assert i in self.train_config, f'{i} not in train config but is required.'
            self.optim = self.train_config.get('optim')
            self.learning_rate = self.train_config.get('learning_rate')
            self.weight_decay = self.train_config.get('weight_decay')
            self.loss = self.train_config.get('loss')
            self.scheduler = self.train_config.get('scheduler')
            self.max_lr = self.train_config.get('max_lr')
            self.epochs = self.train_config.get('epochs')

    def _load_model(self):
        """
        Load model
        """
        #set seed
        if self.model_seed is not None:
            torch.manual_seed=self.model_configseed
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.model_seed)

        if self.model_type == 'w2v2':
            if 'clip_length' in self.data_config:
                clip_length = self.data_config['clip_length']
                if clip_length is not None:
                    trunc = True
                else:
                    trunc = False

                self.data_config['feature_extractor'] = W2V2FeatureExtractor(self.checkpoint, clip_length, truncation=trunc, padding=self.padding)

            self.model = W2V2ForClassification(checkpoint = self.checkpoint, label_dim=self.n_class, pooling_mode=self.pooling_mode,
                                               freeze=self.freeze, weighted=self.weighted, layer=self.layer,  shared_dense=self.shared_dense,
                                               sd_bottleneck=self.sd_bottleneck, clf_bottleneck=self.clf_bottleneck, activation=self.activation,
                                               final_dropout=self.final_dropout, layernorm=self.layernorm)
        elif self.model_type=='ssast':

            if self.mode == 'pretrain':
                self.cluster = (self.num_mel_bins != self.fshape)
                if self.cluster == True:
                    print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(self.num_mel_bins, self.fshape))
                else:
                    print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(self.num_mel_bins, self.fshape))
                
                self.model = ASTModel_pretrain(fshape=self.fshape, tshape=self.tshape, fstride=self.fstride, tstride=self.tstride,
                                            input_fdim =self.num_mel_bins, input_tdim=self.target_length,
                                            model_size=self.model_size, checkpoint=None) 
                print('Checkpoint automatically set to None. Loading pretrained model and continuing training is not yet supported.')
            else:
                assert self.ssast_task in ['ft_cls', 'ft_avgtok']
                self.model = ASTModel_finetune(task=self.ssast_task, label_dim=self.n_class, fshape=self.fshape, tshape=self.tshape,
                                               fstride=self.fstride, tstride=self.tstride, input_fdim=self.num_mel_bins,
                                               input_tdim=self.target_length, model_size=self.model_size, checkpoint=self.checkpoint,
                                               freeze=self.freeze, weighted=self.weighted, layer=self.layer, shared_dense=self.shared_dense,
                                               sd_bottleneck=self.sd_bottleneck, activation=self.activation, final_dropout=self.final_dropout,
                                               layernorm=self.layernorm, clf_bottleneck=self.clf_bottleneck)
        else:
            raise NotImplementedError()

        if self.mode in ['evaluate', 'extract']:
            assert self.finetuned_mdl_path is not None, 'must load a model'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sd = torch.load(self.finetuned_mdl_path, map_location=device)
            self.model.load_state_dict(sd, strict=False)
            
    def _load_data(self):
        """
        Load data from data split root and set up DataLoaders
        """
        if self.mode in ['pretrain', 'finetune']:
            assert '.csv' not in str(self.data_split_root)
        
        if '.csv' not in str(self.data_split_root):
            train_df, val_df, test_df = load_input_data(data_split_root=self.data_split_root, target_labels= self.target_labels,output_dir=self.output_dir, cloud=self.cloud, bucket=self.bucket, val_size=self.val_size, seed=self.data_seed)
       
            if self.debug:
                train_df = train_df[:8]
                val_df = val_df[:8]
                test_df = test_df[:8]
        else:
            test_df = pd.read_csv(str(self.data_split_root))
            test_df = test_df.rename(columns={self.uid_col:'uid'})
            test_df = test_df.set_index('uid')
            test_df=test_df.dropna(subset=self.target_labels)

            if self.debug:
                test_df = test_df[:8]
        
        #set up datasets
        if self.mode in ['pretrain', 'finetune']:
            train_dataset = WaveDataset(train_df, self.input_dir, self.model_type, 'classification', self.train_data_config, self.target_labels, self.bucket)
            val_dataset = WaveDataset(val_df, self.input_dir, self.model_type, 'classification', self.eval_data_config, self.target_labels, self.bucket)
            self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = self.num_workers, collate_fn=collate_clf)
            self.val_loader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=collate_clf)
        eval_dataset = WaveDataset(test_df, self.input_dir, self.model_type, 'classification', self.eval_data_config, self.target_labels, self.bucket)
        self.eval_loader = DataLoader(eval_dataset, batch_size = self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=collate_clf)

    def __call__(self):
        """
        Run Classifier
        """
        #set up Classify object
        if self.mode in ['finetune', 'pretrain']:
            run_clf = Classify(model=self.model, data_type=self.data_type, optimizer=self.optim, learning_rate=self.learning_rate,
                     weight_decay=self.weight_decay, loss_fn = self.loss, scheduler=self.scheduler, max_lr=self.max_lr, 
                     epochs=self.epochs)
        elif self.mode in ['evaluate', 'extract']:
            run_clf = Classify(model=self.model, data_type=self.data_type) #only need model for evaluation
        else:
            raise NotImplemented(f'{self.mode} not implemented.')

        if self.mode == 'finetune':
            print('Finetuning model...')
            run_clf.finetune(dataloader_train = self.train_loader, dataloader_val = self.val_loader, 
                             save=True, save_dir=self.output_dir, cloud=self.cloud['output'], bucket=self.bucket)
        
        if self.mode == 'pretrain':
            print('Pretraining model...')
            assert self.model_type=='ssast', 'Can only pretrain for AST models'
            run_clf.pretrain(dataloader_train=self.train_loader, dataloader_val=self.val_loader, task=self.ssast_task,
                             mask_patch = self.mask_patch, cluster=self.cluster, save=True, save_dir=self.output_dir, 
                             cloud=self.cloud['output'], bucket=self.bucket )

        if self.mode in ['finetune', 'evaluate']:
            run_clf.evaluate(dataloader_eval = self.eval_loader, save=True, save_dir=self.output_dir, 
                               cloud=self.cloud['output'], bucket=self.bucket)

        if self.mode == 'extract':
            if self.model_type == 'ssast':
                self.task = self.ssast_task
            else:
                self.task = None
            run_clf.extract(dataloader=self.eval_loader, embedding_type=self.embedding_type, layer=self.layer,
                            pooling_mode = self.pooling_mode, task=self.task, save=True, save_dir=self.output_dir,
                            cloud=self.cloud['output'], bucket=self.bucket)