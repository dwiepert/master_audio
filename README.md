# ASR Model implementations for NAIP
Includes a handful of model types implemented to get transcriptions (for single audio files - batches not yet implemented), timestamps, and find pauses over a certain duration. Also contains simple metrics like WER/CER

## Installation
To intall naip-asr, follow the instructions below

```
   $ git clone https://github.com/dwiepert/naip_asr.git
   $ cd naip_asr
   $ conda env create -f environment.yml
   $ conda activate naip_asr
   $ pip install .
```

## Model Checkpoints
There are a handful of ways to specify and work with checkpoints
1. Some models can be given a `repo_id` or that is a relative path/model name. For example, Whisper can take `"large"` as a checkpoint to specify which model to use (see [Whisper models](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages) for more. In this case, the model size is the checkpoint). Wav2Vec2 can take some hugging face repo names like `"facebook/wav2vec2-base-960h"`. For some more examples, see [_constants.py](TODO).
2. All models can be given a local directory or file with the pretrained model. This package includes a few ways to download these models to the local directory.
   - Whisper: For local checkpoints, you must download a `.pt` file. These checkpoints can be downloaded using `download_checkpoint_from_url`. If a downloaded model is saved on google cloud storage for space saving, it can also be downloaded using `download_checkpoint_from_gcs`. When giving a local checkpoint to the model, it must be given as the full file path to the `.pt` file (e.g. `local_dir_path\medium.pt`)
   - W2V2: For local checkpoints, you must download the entire checkpoint directory from hugging face using `download_checkpoint_from_hf`. Otherwise, if the directory is saved to the cloud, it can also be downloaded using `download_checkpoint_from_gcs`. Note that `download_checkpoint_from_hf` has pre-built options that can be downloaded ("base", "large", "large-robust", "large-self"), or you can give a `repo_id` and other identifying informationto download checkpoints outside of the pre-built options. The `repo_id` must be from [huggingface.co](https://huggingface.co). When giving a local checkpoint to the model, it must be given as the full path to a downloaded checkpoint directory (e.g. `local_dir_path\w2v2-base-960h`)
   
#### Using download functions
`download_checkpoint_from_hf` takes the following parameters:
- checkpoint: str, Path, directory where the checkpoint file(s) should be saved. Must be full file path to where to SAVE the checkpoint (e.g., `local_dir_path/checkpoint_name`). 
- model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en). This is not required if instead downloading using repo_id.
- model_type: str, specify model type (e.g. whisper, w2v2, ssast). This is not required if instead downloading using repo_id.
- repo_id: str, repo_id in hugging face. This is used if you want to download a model that may not be available to download through specification of model size and type.
- filename: optional filename if downloading a single file instead of directory
- subfolder: str, optional, specify if there is a file in a subdirectory of the hugging face repo

`download_checkpoint_from_url` takes the following parameters:
- checkpoint: str, path to where checkpoint should be stored. This should be the full file path as you use this variable to specify the download target ( .pt for Whisper, .pth for AST Models)
- model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en)
-  model_type: str, specify model type (e.g. whisper, nemo)
- in_memory: bool set to false

`download_checkpoint_from gcs` takes the following parameters:
- checkpoint: str, path to save checkpoint to on local computer. If saving a directory, you only need to give path to the directory where you want to save it. Otherwise, full file path must be given.
- checkpoint_prefix: str, location of checkpoint in bucket. Bucket name should be stripped from this.
- dir: bool, specify if downloading a directory or single file
- project_name: cloud storage project name
- bucket_name: cloud storage bucket name
Note that this function requires access to GCS, which can be given with `gcloud auth application-default login` and `gcloud auth application-default set-quota-project project_name`





## Data Transforms
There are many optional data transform classes. Many of these can be implemented with `WaveDataset` (TODO) through creating a `dataset_config` dictionary with parameters for the transforms.

### Common Transforms 
* `ToNumpy`: Convert waveform in sample to Numpy (not compatible with spectrograms)
* `ToTensor`: Convert targets to a tensor
* `UidToPath`: Given a UID, get full file path and download to local computer if necessary. 
* `UidToWaveform`: Given a UID, load a waveform from either local dir or gcs.

### Audio Transforms
Basic Preprocessing of wav files:
* `Resample`: resample a waveform to `resample_rate`. In `dataset_config`, set the new sampling rate with `resample_rate` and specify whether to resample with librosa using `use_librosa`.
* `ToMonophonic`: convert to monochannel with a reduce function. The `reduce_fn` is not pre-specified. See `WaveDataset` `_getaudiotransforms()` (TODO) for an example of a function. In `dataset_config`, set `monochannel` to True to convert to monochannel.
* `TrimBeginningEndSilence`: trim the beginning and end silence based on `db_threshold`. Set in `dataset_config` with `trim_db`.
* `Truncate`: cut audio to a specified `length` with and optional `offset`. Set clip length to truncate to in `dataset_config` with `clip_length`. `offset` is not set, but this can be changed in `WaveDataset` `__init__` and `_getaudiotransforms()` (TODO) and a key can be added to `dataset_config`.
* `WaveMean`: subtract the mean from the waveform. This is implemented automatically for classification.

Data augmentation of audio files using audiomentations (https://github.com/iver56/audiomentations?tab=readme-ov-file). See below for the variable to set the probablity of the transform in `dataset_config`. Other parameters can be adjusted manually in `WaveDataset` if you would like to change defaults.

* `gauss`: AddGaussianNoise
* `gausssnr`: AddGaussianNoiseSNR
* `alias`: Aliasing
* `bandstop`: BandStopFilter - apparently can aid in preventing overfitting to specific frequency relationships to make models more robust to diverse audio environments and scenarios
* `bitcrush`: BitCrush
* `clipd`: ClippingDistortion
* `gain`: Gain
* `gaint`: GainTransition
* `mp3`: Mp3Compression
* `norm`: Normalize
* `pshift`: PitchShift
* `pinversion`: PolarityInversion
* `tstretch`: TimeStretch
* `tmask`: TimeMask
* `tanh`: TanhDistortion
* `repeat`: RepeatPart
* `reverse`: Reverse
* `room`: RoomSimulator
* `tshift`: Shift

### Spectrogram Tansforms
If working with spectrograms, these are the available transformations:
* `Wav2Fbank`: Convert from wav to spectrogram. Set `target_length`, `melbins`, `tf_co`, and `tf_shift`. Set in `dataset_config` with `target_length`, `num_mel_bins`, `cdo` (boolean, True to use course dropout), `shift` (boolean, True to do an affine shift)
* `FbankNoise`: add random noise to a spectrogram. Specify whether to add noise to a `WaveDataset` with `noise` boolean in `dataset_config`.
* `FreqMaskFbank`: implement frequency masking for SSAST. Set `freqm` in `dataset_config`.
* `NormalizeFbank`: normalize spectrogram. Need to set `norm_mean` (dataset mean) and `norm_std` (dataset standard deviation). These values can be set in `dataset_config` with `dataset_mean` and `dataset_std`
* `TimeMaskFbank`: implement time masking for SSAST. Set `timem` in `dataset_config`.

## Datasets
User defined `WaveDataset` and collate functions to use with torch `DataLoader`. Note that with this dataset, the data is expected to be in the following format: `PATH_TO_INPUT_DIRECTORY/uid/waveform.wav`.
### WaveDataset
Create a WaveDataset for use with the different models available in this package. Compatible with ASR and CLF models, but must be given different dataset config + model_task
Initialize with the following parameters:
* `data`: either an np.array of uids or dataframe with uids as index and annotations in the columns. For classification, must give a dataframe
* `prefix`: location of files to download (compatible with gcs)
* `model_type`: type of model this Dataset will be used with (e.g. w2v2, whisper)
* `model_task`: model task for this Dataset (e.g. asr, classification)
* `dataset_config`: dictionary with transform parameters (see Transforms (TODO: link)). Different for asr and classification. See examples in run_asr and run_clf (TODO: link)
* `target_labels`: str list of targets to extract from data. Can be none only for 'asr'.
* `bucket`: gcs bucket if data is in bucket. If not given, assumes local.  

### Collate functions
Rather than use default collate function, use one of the following when initializing the `DataLoader(Dataset, collate_fn=..., batch_size=...)` depending on the task. `collate_asr` for ASR, `collate_clf` for classification. No parameters need to be initialized. See an example in [ClassificationWrapper(...)](TODO).


## Using ASR Models
See examples of using ASR model in [run_asr.py](TODO).
Please note that Whisper models take a longer time to transcribe an audio file depending on the length of the file as it was originally built to only handle 30s clips.

### Models and functionality
Teo types of models are available to use: Whisper, W2V2. Note that whisper does NOT allow non-words while W2V2 does.

All models have the following method:
- `model.transcribe(audio: either audio path or waveform, return_timestamps: bool, return_pauses: bool, pause_s: float, threshold for long pause in seconds)`: transcribe a single audio file and optionally get timestamps and long pauses
-

#### Whisper
Initialize whisper with the following parameters:
```
from naip_asr.models import WhisperForASR
model = WhisperForASR(checkpoint = checkpoint, model_size=model_size)
```
Where `checkpoint` is either just the model size as a string (e.g. 'base', 'medium') or a full file path to a `.pt` file.

#### W2V2
Initialize W2V2 with the following parameters:
```
from naip_asr.models import W2V2ForASR
model = W2V2ForASR(checkpoint = checkpoint, model_size=model_size)
```
Where `checkpoint` is either a hugging face repo name or a path to a local directory containing the model files.

### Metrics
Word error rate (`wer(...)`) and character error rate (`cer(...)`)are both implemented. Both metrics take the following paramters:
- reference: target transcription
- hypothesis: predicted transcription
- print: boolean indicating whether to print aligned transcription to console (default = False)

## Using Classification Models
See examples of using Classification models in [run_clf.py](TODO). The primary code for running classification models is in [_clf_model_wrapper.py](TODO) and (_classify.py). The arguments that are required in each config dictionary are included in `run_clf.py`. 

### Datasplit
When running classification, generating a datasplit is required. The data splits are expected to be in a single directory with the names `train.csv` and `test.csv` with an optional `validation.csv`. If a datasplit is not already prepared, one can be generated with [generate_datasplit(...)](TODO). See code for required arguments. Note that the metadata for the datasplit must be a csv with the target labels included as columns and at least a uid column (optionally also a subject column). The datasplit can then be generated at either the file or subject level.

### Functionality
This implementation contains many functionality options as listed below. Arguments are set in [run_clf.py](TODO) example.

#### 1. Pretraining
You can pretrain an SSAST model from scratch using the `ASTModel_pretrain` class in [_ast_classification.py](TODO), along with the `pretrain(...)` function of the [`Classify`](TODO) class. 

This mode is triggered by setting `--mode` to 'pretrain' and also specifying which pretraining task to use with `--ssast_task`. The options are 'pretrain_mpc', 'pretrain_mpg', or 'pretrain_joint' which uses both previous tasks.

This function is compatible with data augmentations.

This implementation currently can not continue pretraining from an already pretrained model checkpoint. 

### 2. Finetuning
You can finetune models for classifying speech features using the `ASTModel_finetune` class in [_ast_classification.py](TODO) and [_w2v2_classification](TODO) and the `finetune(...)` function in the [`Classify`](TODO) class. 

This mode is triggered by setting `-m, --mode` to 'finetune' and if using an `ASTModel`, also specifying which finetuning task to use with `--task`. The options are 'ft_cls' or 'ft_avgtok'. See `_cls(x)` and `_avgtok(x)` in [`ASTModel_finetune`](TODO) for more information on how merging is done. 

There are a few different parameters to consider. Firstly, the classification head can be altered to use a different amount of dropout and to include/exclude layernorm. See [`BasicClassifier`](TODO) class for more information. 

Default run mode will also freeze the model and only finetune the classification head. This can be altered with `--freeze`. 

We also include the option to use a different hidden state output as the input to the classification head. This can be specified with `--layer` and must be an integer between 0 and `model.n_states` (or -1 to get the final layer). This works by getting a list of hidden states and indexing using the `layer` parameter. Additionally, you can add a shared dense layer prior to the classification head(s) by specifying  `--shared_dense` along with `--sd_bottleneck` to designate the output size for the shared dense layer. Note that the shared dense layer is followed by ReLU activation. Furthermore, if `shared_dense` is False, it will create an Identity layer so as to avoid if statements in the forward loop. 

Classification head(s) can be implemented in the following manner:
1. Specify `--clf_bottleneck` to designate output for initial linear layer 
2. Give `label_dims` as a list of dimensions or a single int. If given as a list, it will make a number of classifiers equal to the number of dimensions given, with each dimension indicating the output size of the classifier (e.g. [2, 1] will make a classifier with an output of (batch_size, 2) and one with an output of (batch_size, 1). The outputs then need to be stacked by columns to make one combined prediction). In order to do this in [`run_clf.py`](TODO), you must give a label_txt in the following format: split labels with a ',' to specify a group of features to be fed to one classifier; split with a new line '/n' to specify a new classifier. Note that `args.target_labels` should be a flat list of features, but `args.label_groups` should be a list of lists. 

There are data augmentation transforms available for finetuning. 

Finally, we added functionality to train an additional parameter to learn weights for the contribution of each hidden state (excluding the final output, i.e. hidden_states[:-1]) to classification. The weights can be accessed with `MODELCLASS.weightsum`. This mode is triggered by setting `--weighted` to True. If initializing a model outside of the run function, it is still triggered with an argument called `weighted`. 

### 3. Evaluation only
If you have a finetuned/pretrained model and want to evaluate it on a new data set, you can do so by setting `--mode` to 'evaluate'. You must then also specify a `--finetuned_mdl_path` to load in. The datasplit directory `--data_split_root` must then be given as a `.csv`

It is expected that there is an `args.pkl` file in the same directory as the finetuned model to indicate which arguments were used to initialize the finetuned model. This implementation will load the arguments and initialize/load the finetuned model with these arguments. If no such file exists, it will use the arguments from the current run, which could be incompatible if you are not careful. 


### 4. Embedding extraction.
We implemented multiple embedding extraction methods. The implementation is a function within each model called `extract_embedding(..)`, which is called on batches instead of the forward function. 

Embedding extraction is triggered by setting `--mode` to 'extract'. 

You must also consider where you want the embeddings to be extracted from. The options are as follows:
1. From the output of a hidden state? Set `embedding_type` to 'pt'. Can further set an exact hidden state with the `layer` argument. By default, it will use the layer specified at the time of model initialization. The model default is to give the last hidden state run through a normalization layer - ind 13, so the embedding is this output merged to be of size (batch size, embedding_dim). If using an AST model, it will also automatically use the merging strategy defined by the task set at the time of model initialization, but this can be changed at the time of embedding extraction by redefining `task` in `extract_embedding` with either 'ft_cls' or 'ft_avgtok'. This functionality is only for AST models. 
2. After weighting the hidden states? Set `embedding_type` to 'wt'. This version requires that the model was initially finetuned with  `weighted` set to True.
3. After a shared dense layer. Set `embedding_type` to 'st'. This version requires that the model was initially finetuned with `shared_dense` set to True.
4. From a layer in the classification head that has been finetuned? Set `embedding_type` to 'ft'. This version requires specification of `pooling_mode` to merge embeddings if there are multiple classifiers. It only accepts "mean" or "sum" for merging, and if nothing is specified it will use the pooling_mode set with the model. It will always return the output from the first dense layer in the classification head, prior to any activation function or normalization. 

Brief note on target labels:
Embedding extraction is the only mode where target labels are not required. You can give a csv with only uid names  it will still function and extract embeddings.
