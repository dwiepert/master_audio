import torch
import torch.nn.functional
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    '''
    Simple audio dataset
    '''
    
    def __init__(self, annotations_df, target_labels, transform):
        '''
        Initialize dataset with dataframe, target labels, and list of transforms

        '''
        
        self.annotations_df = annotations_df
        self.transform = transform
        self.target_labels = target_labels
        
    def __len__(self):
        
        return len(self.annotations_df)
    
    def __getitem__(self, idx):
        '''
        Run transformation
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx]
        targets = self.annotations_df[self.target_labels].iloc[idx].values
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }
        
        return self.transform(sample)