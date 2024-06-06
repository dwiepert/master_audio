import json
import os
from pathlib import Path
import tempfile
from typing import Union, Tuple

import numpy as np
import torch 
from torch.utils.data import  DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from naip_asr.io import upload_to_gcs

class Classify:
    """
    Object for classification, takes in a model and training parameters and has options for pretraining, finetuning, evaluating, and extracting embeddings from a model

    :param model: classification model
    :param data_type:str, specify data type used by the model (w2v2 = 'waveform', ssast = 'fbank') (default = 'waveform')
    :param optimizer: str, specify which optimizer to use (default = 'adamw', compatible with 'adam' or 'adamw')
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay for adamw optimizer
    :param loss_fn: str, specify which loss functino to use (default = 'BCE', compatible with 'BCE' or 'MSE')
    :param scheduler: str, specify scheduler (default = 'onecycle', compatible with None or 'onecycle')
    :param max_lr: float, max learning rate for scheduler
    """
    def __init__(self, model, data_type:str = 'waveform', optimizer: str ='adamw', learning_rate: float = 0.001, weight_decay: float = 0.0001,
             loss_fn: str='BCE',scheduler: str='onecycle', max_lr: float = 0.01,
             epochs: int =10):
        self.model = model
        self.data_type = data_type
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.scheduler_type = scheduler
        self.max_lr = max_lr
        self.epochs = epochs

        self._set_optimizer()
        self._set_loss()

    
    def _set_optimizer(self):
        """
        Initialize the optimizer based on str name
        """
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],lr=self.learning_rate)
        elif self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f'{self.optimizer_type} is not an implemented optimizer.')
    
    def _set_loss(self):
        """
        Initialize the loss function based on str nme 
        """
        if self.loss_fn == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_fn == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'{self.loss_fn} is not an implemented loss function.')
        
    def _set_scheduler(self, training_dataset_size):
        """
        Initialize the scheduler based on str name
        """
        if self.scheduler_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.ax_lr, steps_per_epoch=training_dataset_size, epochs=self.epochs)
        
        else:
            self.scheduler = None
    
    def pretrain(self, dataloader_train:DataLoader, dataloader_val:DataLoader, save_dir: Union[str,Path], 
                 pretrain_task: str = 'pretrain_joint', mask_patch:int = 400, cluster: bool = False,
                 save: bool = True, cloud=False, bucket=None):
        """
        Pretrain an AST Model. This function is only compatible with SSAST ASTModel_pretrain class. 

        :param dataloader_train: torch DataLoader containing training data
        :param dataloader_val: torch DataLoader containing validation data
        :param save_dir: either str or Path object for the directory to save output to
        :param pretrain_task: pretraining task for AST model (one of ['pretrain_mpg', 'pretrain_mpc', 'pretrain_joint'], default = 'pretrain_joint')
        :param mask_patch: how many patches to mask (used only for ssl pretraining) (default = 400)
        :param cluster: boolean, indicate whether to use cluster masking (automatic if num_mel_bins and fshape are not equivalent, default = False)
        :param save: boolean, indicate whether to save output
        :param cloud: boolean, indicate whether saving to cloud
        :param bucket: gcs bucket object or None
        :return: None, access ClassifyObject.model for the trained model
        """

        assert 'ASTModel_pretrain' in str(self.model.__class__) , 'Only compatible with AST Pretrainable Models'
        if save:
            save_dir = Path(save_dir).absolute()
            if not save_dir.exists() and not cloud:
                os.makedirs(save_dir)
            if cloud:
                assert bucket is not None, 'Must have a gcs bucket if saving to cloud.'

        self._set_scheduler(len(dataloader_train))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        for e in range(self.epochs):
            training_loss = list()
            training_acc = list()
            #t0 = time.time()
            self.model.train()
            for batch in tqdm(dataloader_train):
                x = batch[self.data_type]
                targets = batch['targets']
                x, targets = x.to(device), targets.to(device)
                self.optimizer.zero_grad()
            
                if pretrain_task == 'pretrain_mpc':
                    acc, loss = self.model(x, pretrain_task, mask_patch=mask_patch, cluster=cluster)
                    # this is for multi-gpu support, in our code, loss is calculated in the model
                    # pytorch concatenates the output of each gpu, we thus get mean of the losses of each gpu
                    acc, loss = acc.mean(), loss.mean()
                # if pretrain with generative objective
                elif pretrain_task == 'pretrain_mpg':
                    loss = self.model(x, pretrain_task, mask_patch=mask_patch, cluster=cluster)
                    loss = loss.mean()
                    # dirty code to make the code report mse loss for generative objective
                    acc = loss
                # if pretrain with joint discriminative and generative objective
                elif pretrain_task == 'pretrain_joint':
                    acc, loss1 = self.model(x, 'pretrain_mpc', mask_patch=mask_patch, cluster=cluster)
                    acc, loss1 = acc.mean(), loss1.mean()
                    loss2 = self.model(x, 'pretrain_mpg', mask_patch=mask_patch, cluster=cluster)
                    loss2 = loss2.mean()
                    loss = loss1 + 10 * loss2
                else:
                    raise NotImplementedError(f'{pretrain_task} not implemented for pretraining. Choose one of: pretrain_mpc, pretrain_mpg, pretrain_joint')

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                training_loss.append(loss.detach().cpu().item())
                training_acc.append(acc.detach().cpu().item())

            if e % 10 == 0 or e == self.epochs-1:
                #SET UP LOGS
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()
                else:
                    lr = self.learning_rate
                logs = {'epoch': e, 'optim':self.optimizer_type, 'lr': lr}

                logs['training_loss_list'] = training_loss
                training_loss = np.array(training_loss)
                logs['running_loss'] = np.sum(training_loss)
                logs['training_loss'] = np.mean(training_loss)

                print('RUNNING LOSS', e, np.sum(training_loss) )
                print(f'Training loss: {np.mean(training_loss)}')

                logs['training_acc_list'] = training_acc
                training_acc = np.array(training_acc)
                logs['training_acc'] = np.mean(training_acc)
            
                print(f'Training acc: {np.mean(training_acc)}')

                if dataloader_val is not None:
                    print("Validation start")
                    validation_loss, validation_acc = self._validation_mask(dataloader_val, pretrain_task=pretrain_task, cluster=cluster, mask_patch=mask_patch)

                    logs['val_loss_list'] = validation_loss
                    validation_loss = np.array(validation_loss)
                    logs['val_running_loss'] = np.sum(validation_loss)
                    logs['val_loss'] = np.mean(validation_loss)
                    
                    print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                    print(f'Validation loss: {np.mean(validation_loss)}')

                    logs['val_acc_list'] = validation_acc
                    validation_acc = np.array(validation_acc)
                    logs['val_acc'] = np.mean(validation_acc)

                    print(f'Validation acc: {np.mean(validation_acc)}')
                
                #SAVE LOGS
                if save:
                    self._save_model(e=e, logs=logs, save_dir=save_dir, cloud=cloud, bucket=bucket)

    def _save_model(self, e:int, logs: dict, save_dir: Union[str, Path], cloud: bool = False, bucket = None):
        """
        Save trained model and logs

        :param e: int, current epoch 
        :param logs: dictionary of model logs
        :param save_dir: directory to save to
        :param cloud: bool, indicate whether to save to cloud
        :param bucket: gsc bucket object

        :return: None, saves model
        """
        print(f'Saving epoch {e}')
        if cloud:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                json_string = json.dumps(logs)
                logs_path = temp_dir / f'logs_epoch{e}.json'
                with open(logs_path, 'w') as outfile:
                    json.dump(json_string, outfile)
            
                #SAVE CURRENT MODEL
                
                mdl_path =temp_dir/ f'mdl_epoch{e}.pt'
                torch.save(self.model.state_dict(), mdl_path)
                
                optim_path = temp_dir/ f'optim_epoch{e}.pt'
                torch.save(self.optimizer.state_dict(), optim_path)
                upload_to_gcs(gcs_prefix=save_dir, path=logs_path, bucket=bucket)
                upload_to_gcs(gcs_prefix=save_dir, path=mdl_path, bucket=bucket)
                upload_to_gcs(gcs_prefix=save_dir, path=optim_path, bucket=bucket)

        else:
            json_string = json.dumps(logs)
            logs_path = save_dir / f'logs_epoch{e}.json'
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
        
            #SAVE CURRENT MODEL
            
            mdl_path =save_dir / f'mdl_epoch{e}.pt'
            torch.save(self.model.state_dict(), mdl_path)
            
            optim_path = save_dir / f'optim_epoch{e}.pt'
            torch.save(self.optimizer.state_dict(), optim_path)

    def _validation_mask(self, dataloader_val: DataLoader, pretrain_task: str = 'pretrain_joint', cluster:bool = False, mask_patch: int = 400):
        '''
        Validation loop for pretraining with SSAST
        :param dataloader_val: dataloader object with validation data
        :param pretrain_task: pretraining task for AST model (one of ['pretrain_mpg', 'pretrain_mpc', 'pretrain_joint'], default = 'pretrain_joint')
        :param mask_patch: how many patches to mask (used only for ssl pretraining) (default = 400)
        :param cluster: boolean, indicate whether to use cluster masking (automatic if num_mel_bins and fshape are not equivalent, default = False)
        :return validation_loss: list with validation loss for each batch
        :return validation_acc: list with validation accuracy for each batch
        '''
        validation_loss = list()
        validation_acc = list()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader_val):
                x = batch[self.data_type]
                targets = batch['targets']
                x, targets = x.to(device), targets.to(device)
                # always use mask_patch=400 for evaluation, even the training mask patch number differs.
                if pretrain_task == 'pretrain_mpc':
                    acc, nce = self.model(x, pretrain_task, mask_patch=mask_patch, cluster=cluster)
                    validation_loss.append(nce.detach().cpu().item())
                    validation_acc.append(acc.detach().cpu().item())
                elif pretrain_task == 'pretrain_mpg':
                    mse = self.model(x, pretrain_task, mask_patch=mask_patch, cluster=cluster)
                    # this is dirty code to track mse loss, A_acc and A_nce now track mse, not the name suggests
                    validation_loss.append(mse.detach().cpu().item())
                    validation_acc.append(mse.detach().cpu().item())
                elif pretrain_task == 'pretrain_joint':
                    acc, _ = self.model(x, 'pretrain_mpc', mask_patch=mask_patch, cluster=cluster)
                    mse = self.model(x, 'pretrain_mpg', mask_patch=mask_patch, cluster=cluster)

                    validation_loss.append(mse.detach().cpu().item())
                    validation_acc.append(acc.detach().cpu().item())

        return validation_loss, validation_acc


    def finetune(self, dataloader_train: DataLoader, dataloader_val: DataLoader, save_dir: Union[str,Path], save: bool = True, cloud=False, bucket=None):
        """
        Finetune a model. Must not be ASTModel_pretrain class

        :param dataloader_train: torch DataLoader containing training data
        :param dataloader_val: torch DataLoader containing validation data
        :param save_dir: either str or Path object for the directory to save output to
        :param save: boolean, indicate whether to save output
        :param cloud: boolean, indicate whether saving to cloud
        :param bucket: gcs bucket object or None
        :return: None, access ClassifyObject.model for the finetuned model
        """
        assert 'ASTModel_pretrain' not in str(self.model.__class__), 'Not compatible with ASTModel_Pretrain.'
        if save:
            save_dir = Path(save_dir).absolute()
            if not save_dir.exists() and not cloud:
                os.makedirs(save_dir)
            if cloud:
                assert bucket is not None, 'Must have a gcs bucket if saving to cloud.'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self._set_scheduler(len(dataloader_train))

        for e in range(self.epochs):
            training_loss = list()
        #t0 = time.time()
            self.model.train()
            for batch in tqdm(dataloader_train):
                if self.data_type == 'waveform':
                    x = torch.squeeze(batch[self.data_type][0], dim=1)
                else:
                    x = batch[self.data_type]
                targets = batch['targets']
                x, targets = x.to(device), targets.to(device)
                self.optimizer.zero_grad()
                o = self.model(x)
                loss = self.criterion(o, targets)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                loss_item = loss.item()
                training_loss.append(loss_item)

            if e % 10 == 0 or e == self.epochs-1:
                #SET UP LOGS
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()
                else:
                    lr = self.learning_rate
                logs = {'epoch': e, 'optim':self.optimizer_type, 'loss_fn': self.loss_fn, 'lr': lr, 'scheduler':self.scheduler_type}
        
                logs['training_loss_list'] = training_loss
                training_loss = np.array(training_loss)
                logs['running_loss'] = np.sum(training_loss)
                logs['training_loss'] = np.mean(training_loss)

                print('RUNNING LOSS', e, np.sum(training_loss) )
                print(f'Training loss: {np.mean(training_loss)}')

                if dataloader_val is not None:
                    print("Validation start")
                    validation_loss = self._validation(dataloader_val)
                   
           
                    logs['val_loss_list'] = validation_loss
                    validation_loss = np.array(validation_loss)
                    logs['val_running_loss'] = np.sum(validation_loss)
                    logs['val_loss'] = np.mean(validation_loss)
                    
                    print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                    print(f'Validation loss: {np.mean(validation_loss)}')
                
                #SAVE LOGS
                if save:
                    self._save_model(e=e, logs=logs, save_dir=save_dir, cloud=cloud, bucket=bucket)
                  

    def _validation(self, dataloader_val: DataLoader) -> list:
        """
        Validation loop for finetuning model

        :param dataloader_val: torch DataLoader with validation data
        :return validation_loss: list of loss for validation set
        """
        validation_loss = list()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader_val):
                if self.data_type == 'waveform':
                    x = torch.squeeze(batch[self.data_type], dim=1)
                else:
                    x = batch[self.data_type]
                targets = batch['targets']
                x, targets = x.to(device), targets.to(device)
                o = self.model(x)
                val_loss = self.criterion(o, targets)
                validation_loss.append(val_loss.item())

        return validation_loss

    def evaluate(self, dataloader_eval: DataLoader,  save_dir: Union[str,Path], 
                 save: bool = True, cloud:bool = False, bucket = None) -> Tuple[torch.tensor, torch.tensor]:
        """
        Evaluate a trained model

        :param dataloader_eval: torch DataLoader with evaluation loader
        :param save_dir: either str or Path object for the directory to save output to
        :param save: boolean, indicate whether to save output
        :param cloud: boolean, indicate whether saving to cloud
        :param bucket: gcs bucket object or None

        :return outputs: outputs as torch.tensor
        :return t: targets as torch.tensor
        """
        if save:
            save_dir = Path(save_dir).absolute()
            if not save_dir.exists() and not cloud:
                os.makedirs(save_dir)
            if cloud:
                assert bucket is not None, 'Must have a gcs bucket if saving to cloud.'


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = []
        t = []
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader_eval):
                if self.data_type == 'waveform':
                    x = torch.squeeze(batch[self.data_type], dim=1)
                else:
                    x = batch[self.data_type]
                x = x.to(device)
                targets = batch['targets']
                targets = targets.to(device)
                o = self.model(x)
                outputs.append(o)
                t.append(targets)

        outputs = torch.cat(outputs).cpu().detach()
        t = torch.cat(t).cpu().detach()
        
        # SAVE PREDICTIONS AND TARGETS
        if save: 
            print(f'Saving predictions...')
            if cloud:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    prediction_path = temp_dir / 'predictions.pt'
                    target_path = temp_dir / 'targets.pt'
                    torch.save(outputs, prediction_path)
                    torch.save(t, target_path)
                    
                    upload_to_gcs(gcs_prefix=save_dir, path=prediction_path, bucket=bucket)
                    upload_to_gcs(gcs_prefix=save_dir, path=target_path, bucket=bucket)

            else:
                prediction_path = save_dir / 'predictions.pt'
                target_path = save_dir / 'targets.pt'
                torch.save(outputs, prediction_path)
                torch.save(t, target_path)

        return outputs, t
    
    def extract(self, dataloader: DataLoader, save_dir: Union[str,Path], 
                embedding_type='ft',layer=-1, pooling_mode='mean', embedding_task: str = None, 
                save: bool = True, cloud:bool = False, bucket = None) -> np.ndarray:
        
        """
        Extract embeddings from a trained/finetuned model

        :param dataloader: torch DataLoader with audio to extract embeddings
        :param save_dir: either str or Path object for the directory to save output to
        :param embedding_type: specify whether embeddings should be extracted from classification head (ft), base pretrained model (pt), weighted sum (wt),or shared dense layer (st) (default = 'ft')
        :param layer: int, layer to extract from (default = -1 (last layer))
        :param save: boolean, indicate whether to save output (default = True)
        :param cloud: boolean, indicate whether saving to cloud (default = False)
        :param bucket: gcs bucket object or None (default = None)

        :return embeddings: numpy array of embeddings
        """

        if 'ASTModel' in str(self.model.__class__):
            assert embedding_task is not None, 'Must give an embedding task for AST models'
        elif 'W2V2' in str(self.model__class__):
            embedding_task = None #must have an empty task for w2v2
        else:
            raise NotImplementedError()

        if save:
            save_dir = Path(save_dir).absolute()
            if not save_dir.exists() and not cloud:
                os.makedirs(save_dir)
            if cloud:
                assert bucket is not None, 'Must have a gcs bucket if saving to cloud.'

        embeddings = np.array([])

        # send to gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        uid = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader):
                uid.append(batch['uid'][0])
                if self.data_type == 'waveform':
                    x = torch.squeeze(batch[self.data_type], dim=1)
                else:
                    x = batch[self.data_type]
                x = x.to(device)
                if embedding_task is not None:
                    e = self.model.extract_embedding(x, embedding_type=embedding_type, layer=layer, task=embedding_task, pooling_mode=pooling_mode)
                else:
                    e = self.model.extract_embedding(x, embedding_type=embedding_type,layer=layer, pooling_mode=pooling_mode)
                e = e.cpu().numpy()
                if embeddings.size == 0:
                    embeddings = e
                else:
                    embeddings = np.append(embeddings, e, axis=0)
        
        df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index = uid)
        
        if save: 
            if cloud:
                print('Saving embeddings...')
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    try:
                        embd_path = temp_dir / 'embeddings.pqt'
                        df_embed.to_parquet(path=embd_path, index=True, engine='pyarrow') #TODO: fix
                    except:
                        print('Unable to save as pqt, saving instead as csv')
                        embd_path = temp_dir /'embeddings.csv'
                        df_embed.to_csv(embd_path, index=True)
                    
                    upload_to_gcs(save_dir, embd_path, bucket)

            else:
                try:
                    embd_path = save_dir / 'embeddings.pqt'
                    df_embed.to_parquet(path=embd_path, index=True, engine='pyarrow') #TODO: fix
                except:
                    print('Unable to save as pqt, saving instead as csv')
                    embd_path = save_dir /'embeddings.csv'
                    df_embed.to_csv(embd_path, index=True)
        
        return embeddings

        
