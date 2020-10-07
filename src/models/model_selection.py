from tensorflow import keras
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

# project packages
from src.preprocess.dataset import *
from src.models import classifier
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# Define variables
batch_size = 32
np.random.seed(330)

def cross_validate(dataset_df, subjID_idx, classifier, params, cv=5):
    '''
    Performs k-fold cross-validation
    '''
    
    # Define variables
    k_fold = cv
    kf = KFold(n_splits = k_fold)
    methods = 'train','val'
    
    # initialize cv result dict
    results = {}
    methods = 'train','val'
    for kk in range(1,k_fold+1):
        results['fold%i' %(kk)] = {}
        for method in methods:
            results['fold%i' %(kk)][method] = []
        
    # Start cross-validation
    cv = 0
    for train_idx, val_idx in kf.split(subjID_idx):
        cv += 1
        print('CV Fold: ',cv)
        train_X, train_Y = query_dataset(dataset_df, train_idx)
        val_X, val_Y = query_dataset(dataset_df, val_idx)
        print('Training.....')
        model = classifier(train_X, train_Y, params['L2'], 
                           params['dropout'],  
                           params['lr'], 
                           epochs=20, batch_size=batch_size)
        print('completed!')
        train_acc = model.evaluate(train_X,train_Y)[1]
        val_acc = model.evaluate(val_X,val_Y)[1]
        results['fold%i' %(cv)]['train'].append(train_acc)
        results['fold%i' %(cv)]['val'].append(val_acc)
        
    return results
    

def grid_search_cv(dataset_df, subjID_idx, classifier, param_grid, cv=5, random_n=None):
    '''
    Performs grid search along with kfold cv
    '''
    
    if random_n:
        random.seed(47)
        param_grid = random.sample(list(ParameterGrid(param_grid)), random_n)
    else:
        param_grid = list(ParameterGrid(param_grid))
    
    results = defaultdict(dict)
    
    # Start grid search
    for n, params in enumerate(param_grid):
        print('\nModel %i' %(n))
        results['model%i' %(n)] = cross_validate(dataset_df, 
                                                 subjID_idx, 
                                                 classifier, 
                                                 params, 
                                                 cv=cv)

    return results, param_grid

if __name__ == '__main__':
    
    # Load data
    path = '/home/climbach/approach-retreat/data/raw/00a-ROI316_withShiftedSegments.pkl'
    dataset = Dataset(path)
    dataset.load()
    dataset.train_test_split_sid()
    
    # Re-organize the raw data
    dataset_df = organize_dataset(dataset.data)

    # Define the parameter grid for grid search
    params = {'L2':[0,0.001,0.003,0.01,0.03],
              'dropout':[0, 0.1, 0.2, 0.3, 0.4],
              'lr':[0.001,0.003,0.006]}
    
    results, param_grid = grid_search_cv(dataset_df,
                                         dataset.train_sid,
                                         classifier,
                                         params)
    
    # Save the results and parameter grid in as pickle file
    with open('/home/climbach/approach-retreat/models/grid_search_results.pkl',"wb") as f:
        pkl.dump([results,param_grid],f)