import torch

class ToTensor(object):
    '''
    Convert labels to a tensor rather than ndarray
    '''
    def __call__(self, sample):
        
        targets = sample['targets']
        sample['targets'] = torch.from_numpy(sample['targets']).type(torch.float32)
        
        return sample