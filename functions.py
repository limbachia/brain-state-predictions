import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
from functools import partial
from sklearn.model_selection import ParameterGrid
import random
import pickle as pkl
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold


k_fold = 5
k_seed = 42
TR = 7
np.random.seed(330)
methods = 'train', 'val'


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
            target_list = [target[:7, idx_trial], target[7:, idx_trial]]
            label_list = [0, 1]
            
            dfs.append(pd.DataFrame({"participant":participant_list,
                                     "trial":trial_list,
                                     "data":data_list,
                                     "label":label_list,
                                     "target":target_list}))
    return pd.concat(dfs)

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

def get_data_labels(dataset):
    X_list = []
    Y_list = []
    for pid, value in dataset.items():
        
        data = value['data'] #(14 time points, ROI/VOX, trial)
        _, _, num_trials = data.shape
        
        for idx_trial in range(num_trials):
            # separate approach/retreat and stack them together 
            X_list += [data[:7, :, idx_trial], data[7:, :, idx_trial]]
            Y_list += [np.zeros(7), np.ones(7)]
            
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y



# Classifier function 
def classifier(train_X, train_Y, 
                l2, dropout, kernel_size, 
                strides, lr, epochs=20, 
                batch_size=32):
    np.random.seed(42)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                      kernel_regularizer=regularizer,
                      dropout=dropout,recurrent_dropout=dropout)
    if strides > 1:
        padding = "valid"
    else:
        padding = "same"

    model = keras.models.Sequential([
                              CustomGRU(16,return_sequences=True,input_shape=[None, train_X.shape[-1]]),
                              CustomGRU(16,return_sequences=True),
                              CustomGRU(16),
                              keras.layers.Dense(1,activation='sigmoid')
                              ])
    ### The following model with top convolutional layer works better, but in ada gpu the conv operation is
    ### giving error. Hence the top layer is replaced with a customized GRU layer (see the above model).
    #model = keras.models.Sequential([
    #                          keras.layers.Conv1D(filters=7, 
    #                                              kernel_size=kernel_size,
    #                                              strides=strides, 
    #                                              padding=padding,
    #                                              input_shape = [None, train_X.shape[-1]]),
    #                          CustomGRU(16,return_sequences=True),
    #                          CustomGRU(16),
    #                          keras.layers.Dense(1,activation='sigmoid')
    #                          ])
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['acc'])
    model.fit(train_X,train_Y,epochs=epochs, validation_split=0.2,batch_size=batch_size,verbose=0)
    return model


## Permutation function
def permute(train_X, train_Y, val_X, val_Y, params, k_perm=0):
    results_rand = {}
    for method in methods:
        results_rand[method] = []

    for i_perm in range(k_perm):
        if i_perm % 100 == 0:
            print("--------perm = %d--------" %(i_perm))
        '''
        permute training labels,
        retain test labels
        https://www.ncbi.nlm.nih.gov/pubmed/19070668
        '''
        train_Y_perm = train_Y.copy()
        np.random.shuffle(train_Y_perm)

        model = classifier(train_X, train_Y_perm, params['l2'], 
                      params['dropout'], 
                      params['kernel_size'], 
                      params['strides'], 
                      params['lr'])

        train_acc = model.evaluate(train_X, train_Y_perm)[1]
        val_acc = model.evaluate(val_X, val_Y)[1]

        if i_perm % 100 == 0:
            print('permuted train acc: %.3f' %(train_acc))
            print('permuted val acc: %.3f' %(val_acc))

        results_rand['train'].append(train_acc)
        results_rand['val'].append(val_acc)

    return results_rand

## Temporal accuracy
def temporal_accuracy(train_X, train_Y, val_X, val_Y, params):
    results_temp = {}
    for ts in range(1,TR+1):
        results_temp['TR%i' %(ts)] = {}
        for method in methods:
            results_temp['TR%i' %(ts)][method] = []
            
    for ts in range(1,TR+1):
        print('Timestep %i' %(ts))
        train_X_ts = train_X[:,:ts,:]
        val_X_ts = val_X[:,:ts,:]
        print('Training.....')
        model = classifier(train_X_ts, train_Y, params['l2'], 
                           params['dropout'], 
                           params['kernel_size'], 
                           params['strides'], 
                           params['lr'])
        print('completed!')
        train_acc = model.evaluate(train_X_ts,train_Y)[1]
        val_acc = model.evaluate(val_X_ts,val_Y)[1]
        
        results_temp['TR%i' %(ts)]['train'].append(train_acc)
        results_temp['TR%i' %(ts)]['val'].append(val_acc)
        
    return results_temp
    
## Cross-validation function
def cross_val_scores(dataset_df, nonTest_idx, params, temp_acc = True, k_perm = 0):
    '''
    outputs the following for every fold
        results: overall accuracy
        results_temp: temporal accuracy
        results_rand: null accuracy distribution
    '''
    # Initialize results dict to store all accuracy values
    results = {}
    for kk in range(1,k_fold+1):
        results['fold%i' %(kk)] = {}
        for method in methods:
            results['fold%i' %(kk)][method] = []
            
    # Initialize results_rand dict to store accuracies of permuted models
    results_rand = {}
        
    # Initialize results_temp dict to store temporal accuracies
    results_temp = {}
    
    # Split the training set into k_folds
    kf = KFold(n_splits = k_fold)
    cv = 0
    
    for train_idx, val_idx in kf.split(nonTest_idx):
        cv += 1
        print('\nCV Fold: ',cv)
        train_X, train_Y = query_dataset(dataset_df, train_idx)
        val_X, val_Y = query_dataset(dataset_df, val_idx)
        print('Training.....')
        model = classifier(train_X, train_Y, params['l2'], 
                           params['dropout'], 
                           params['kernel_size'], 
                           params['strides'], 
                           params['lr'])
        print('completed!')
        train_acc = model.evaluate(train_X,train_Y)[1]
        val_acc = model.evaluate(val_X,val_Y)[1]

        results['fold%i' %(cv)]['train'].append(train_acc)
        results['fold%i' %(cv)]['val'].append(val_acc)
        
        if temp_acc:
            results_temp['fold%i' %(cv)] = temporal_accuracy(train_X, train_Y, val_X, val_Y, params)
        if k_perm != 0:
            results_rand['fold%i' %(cv)] = permute(train_X, train_Y, val_X, val_Y, params, k_perm=k_perm)

    return results, results_temp, results_rand