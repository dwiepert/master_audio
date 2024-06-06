
def collate_asr(batch):
    """
    Custom collate fn for ASR (when using dataloader) to keep data type of each item
    """
    #options are [uid]
    uid = [item['uid'] for item in batch]
    out_batch = {'uid':uid}

    if 'sample_rate' in batch[0]:
        sr = [item['sample_rate'] for item in batch]
        out_batch['sample_rate'] = sr
    
    waveform = [item['waveform'] for item in batch]
    out_batch['waveform'] = waveform
    
    return out_batch
   
