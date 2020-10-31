from sklearn.model_selection import ShuffleSplit
import logging
import pickle as pkl
import pandas as pd
import numpy as np
import copy

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
        self.data = None
        self.train_idx = None
        self.test_idx = None
    
    def load(self):
        if self.data is None:
            with open(self.path,'rb') as f:
                self.data = pkl.load(f)
    
    def sid(self):
        if self.data is None:
            raise ValueError("Data not loaded. Load the data using load() method")
    
        return list(self.data.keys())
    
    def train_test_split_sid(self, n_splits=50, 
                         train_size=0.7, random_state=0):
        '''
        Returns train and test sids
        '''
        if self.data is None:
            raise ValueError("Data not loaded. Load the data using load() method")
            
        rs = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
        all_participants = self.sid()
        for train_idx, test_idx in rs.split(self.sid()):
            pass
        
        self.train_idx = train_idx
        self.test_idx = test_idx
    
    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded. Load the data using load() method")
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
            label_list = [1, 0]
            
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
    
    Parameters
    ----------
    dataset_df: pandas.DataFrame
                Data-frame generated by the `organize_dataset()`
    idx_participants: numpy.1darray or a list
                        indexes of participants in dataset_df
                        
                        
    Returns
    -------
    tuple of numpy.ndarray,
        X, y
    """
    participants = dataset_df.participant.unique()[idx_participants]
    rows = dataset_df[dataset_df.participant.isin(participants)]
    
    X = np.stack(rows.data.values, 
                 axis=0).astype(np.float64) 
    Y = np.stack(rows.label.values, 
                 axis=0).astype(np.float64)
    return X, Y


def selective_segments(dataset,TP='all'):
    '''
    Input
    -----
    dataset: dataset dictionary (saved as pkl file)
    TP: timepoint
        int (0,1,2,3,4,or 5) or string (only 'all' is allowed))
        
    Return
    ------
    SEGMENTS: dataset dictionary only with those segment 
             indicated by TP. If TP = 'all', returns
             segments for all TPs
    '''
    TP_map = {0:5, 1:4, 2:3, 3:2, 4:1, 5:0}
    SEGMENTS = copy.deepcopy(dataset)
    if TP == 'all':
        for subj in SEGMENTS:
            data = SEGMENTS[subj]['data']
            SEGMENTS[subj]['data'] = data.reshape((data.shape[:2])
                                                  +(data.shape[-2]*data.shape[-1],))
    elif TP in list(range(6)):
        for subj in SEGMENTS:
            data = SEGMENTS[subj]['data']
            SEGMENTS[subj]['data'] = data[:,:,:,TP_map[TP]]
    else:
        raise ValueError('int:0,1,2,3,4,5 and str:"all" are only valid TP values')
        
    return SEGMENTS