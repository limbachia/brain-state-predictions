from tensorflow import keras
from functools import partial
from collections import defaultdict
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# project packages
from src.preprocess.dataset import *
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# Define variables
batch_size = 32
np.random.seed(330)

def classifier(train_X, train_Y, 
               l2, dropout, lr, 
               epochs=20, 
               batch_size=32,
               seed=42):
    '''
    A sequence-to-vector GRU model.
    
    Parameters
    ----------
    train_X: numpy array (batch x time X rois)
    train_y: numpy array (batch x 1)
    l2: L2 regularization (float)
    dropout: dropout fraction (float)
    batch_size: int, default = 32
    seed: int, default = 42
    
    Returns
    -------
    model: trained model
    '''
    tf.random.set_seed(seed)
    #np.random.seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                        kernel_regularizer=regularizer,
                        dropout=dropout,recurrent_dropout=dropout)
    
    model = keras.models.Sequential([
                              CustomGRU(16,return_sequences=True,
                                        input_shape=[None, 
                                                     train_X.shape[-1]]),
                              CustomGRU(16,return_sequences=True),
                              CustomGRU(16,return_sequences=True),
                              keras.layers.TimeDistributed(keras.layers.Dense(1,activation='sigmoid'))
                              ])
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,metrics=['acc'])
    model.fit(train_X,train_Y,epochs=epochs,
              validation_split=0.2,batch_size=batch_size,verbose=0)

    return model

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
    
    
class MyGridSearchCV:
    """ Grid search of the parameter space to find best parameters
    
    This class contains `fit` and that is compatible with the sklearn model classes. 
    the `best_params` attribute returns the best parameters.
    """
    def __init__(self):
        self.best_params = None
        self.results = None
        self.param_grid = None
        
    def fit(self,dataset_df, subjID_idx, classifier, param_grid, cv=5, n_models=None):
        
        if n_models:
            random.seed(47)
            param_grid = random.sample(list(ParameterGrid(param_grid)), n_models)
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

        self.results, self.param_grid = results, param_grid

def permute(train_X, train_Y, val_X, val_Y, params, k_perm=1000):
    '''
    Inputs
    ------
    train_X: (train_batch_size x time x features)
    train_Y: (train_batch_size,)
    val_X: (val_batch_size x time x features)
    val_Y: (val_batch_size,)
    params: dict(L2: 0.003, dropout:0.3, ;lr:0.001)
    k_perm: number of times training labels should be shuffled
            and trained. For example, if k_perm = 100,
            training labels will be shuffled 100 times, and 
            after each shuffle trainig will take place on the 
            training set and testing will take place on the 
            non-shuffled validation set.
            
    Returns
    -------
    null_accu_dist: validation set accuracy array of size (k_perm,).
    '''
    results_rand = {}
    for method in 'train val'.split():
        results_rand[method] = []

    tf.random.set_seed(330)
    for i_perm in range(k_perm):
        if i_perm % (k_perm//10) == 0:
            print("--------perm = %d--------" %(i_perm))
        '''
        permute training labels,
        retain test labels
        https://www.ncbi.nlm.nih.gov/pubmed/19070668
        '''
        train_Y_perm = train_Y.copy()
        np.random.shuffle(train_Y_perm)

        model = classifier(train_X,train_Y_perm,
                           params['L2'],
                           params['dropout'],
                           params['lr'])

        train_acc = model.evaluate(train_X, train_Y_perm)[1]
        val_acc = model.evaluate(val_X, val_Y)[1]

        if i_perm % (k_perm//10) == 0:
            print('permuted train acc: %.3f' %(train_acc))
            print('permuted val acc: %.3f' %(val_acc))

        results_rand['train'].append(train_acc)
        results_rand['val'].append(val_acc)

    return results_rand
