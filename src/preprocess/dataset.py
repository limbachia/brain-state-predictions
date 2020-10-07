import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

class Dataset:
    '''
    Dataset object
        attributes:
            path: dataset path
            
        methods:
            sid: get subject IDs
            load: load the dataset
            len: total number of subjects
    '''
    def __init__(self, path):
        self.path = path
        self.data = self.load()
    
    def load(self):
        with open(self.path,'rb') as f:
            self.data = pkl.load(f)
    
    def sid(self):
        return list(self.data.keys())
    
    def train_test_split_sid(self, n_splits=50, 
                         train_size=0.7, random_state=0):
        '''
        Returns train and test sids
        '''
        
        rs = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
        all_participants = self.sid()
        for train_idx, test_idx in rs.split(self.sid()):
            pass
        
        self.train_idx = train_idx
        self.test_idx = test_idx
    
    def __len__(self):
        return len(list(self.data.keys()))
    
    

def organize_dataset(dataset):
    """ Create a dataframe for each trial of each participant
    dataframe has columns
        - participant
        - trial
        - data : input to the model
        - label : class label for approach/retreat
    Finally concatenate dataframes for all {participants, trials}
    """
    dfs = []
    for pid, value in dataset.items():
        
        data = value['data'] #(14 time points, ROI/VOX, trial)
        target = value['target']
        num_timepoints, _, num_trials = data.shape
        
        for idx_trial in range(num_trials):
            # separate approach/retreat and stack them together 
            participant_list = [pid] * 2
            trial_list = [idx_trial] * 2
            data_list = [data[:7, :, idx_trial], data[7:, :, idx_trial]]
            #target_list = [target[:7, idx_trial], target[7:, idx_trial]]
            label_list = [0, 1]
            
            dfs.append(pd.DataFrame({"participant":participant_list,
                                     "trial":trial_list,
                                     "data":data_list,
                                     "label":label_list}))#,
                                     #"target":target_list}))
            
    dataset_df = pd.concat(dfs)
    return dataset_df


def query_dataset(dataset_df, idx_participants):
    """Queries data and corresponding labels for the list of participants
    as numpy arrays.
    """
    participants = dataset_df.participant.unique()[idx_participants]
    rows = dataset_df[dataset_df.participant.isin(participants)]
    
    X = np.stack(rows.data.values, 
                 axis=0).astype(np.float64) 
    Y = np.stack(rows.label.values, 
                 axis=0).astype(np.float64)
    return X, Y