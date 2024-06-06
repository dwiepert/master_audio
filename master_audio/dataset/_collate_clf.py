import torch

def collate_clf(batch):
    '''
    This collate function is meant for use when initializing a dataloader for classification - pass for the collate_fn argument.
    Only use this version if you are wanting to maintain the waveform information of a batch that has different length tensors rather than
    padding the waveform. Otherwise, use the default collate_fn.
    This function also only accounts for information maintained with the transformations laid out in this script. If more information is added
    to the samples, it needs to be adjusted.
    '''
    uid = [item['uid'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    targets = torch.stack([item['targets'] for item in batch])
    sample = {'uid':uid, 'targets':targets, 'sample_rate':sr}

    if 'waveform' in batch[0]:
        #check 
        len = batch[0]['waveform'].size()
        all_same = [item['waveform'].size() == len for item in batch]
        if all(all_same):
            waveform = torch.stack([item['waveform'] for item in batch])
        else:
            waveform = [item['waveform'] for item in batch]
        sample['waveform'] = waveform

    
    if 'fbank' in batch[0]:
        fbank = torch.stack([item['fbank'] for item in batch])
        sample['fbank'] = fbank
   
    return sample

