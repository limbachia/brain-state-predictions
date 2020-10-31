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
import logging

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
    
    np.random.seed(seed)
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

class MajorityVoteClassifier:
    """ Majority Vote Classifier 
    
    This class contains the `fit` and `predict` methods that are compatible
    with the sklearn model classes.
    """
    
    def __init__(self):
        self.majority_vote = None
        
    def fit(self,X, y):
        self.majority_vote = round(y.mean())
        
    def predict(self, X):
        if self.majority_vote is None:
            raise ValueError("The majority vote classifier has to be trained before making predictions")
        return [self.majority_vote]*len(X)
    
    
def run_majority_vote(X_train, X_test, y_train, y_test):
    """ Use the majority vote to predict survival.
    
    Parameters
    ----------
    X_train: numpy.ndarray
    X_test: numpy.ndarray
    y_train: numpy.ndarray
    y_test: numpy.ndarray
    
    """
    
    logging.info("Running the majority vote classifier")
    
    majority_vote_classifier = MajorityVoteClassifier()
    majority_vote_classifier.fit(X_train, y_train)
    y_test_predictions = majority_vote_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_true=y_test, y_pred=y_test_predictions)
    
    logging.info('The prediction accuracy with the majority vote classifier is {:.1f}%'.format(accuracy*100))
    
    return majority_vote_classifier


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
        logging.info('CV Fold: ',cv)
        train_X, train_Y = query_dataset(dataset_df, train_idx)
        val_X, val_Y = query_dataset(dataset_df, val_idx)
        logging.info('Training.....')
        model = classifier(train_X, train_Y, params['L2'], 
                           params['dropout'],  
                           params['lr'], 
                           epochs=20, batch_size=batch_size)
        logging.info('completed!')
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
            logging.info('\nModel %i' %(n))
            results['model%i' %(n)] = cross_validate(dataset_df, 
                                                     subjID_idx, 
                                                     classifier, 
                                                     params, 
                                                     cv=cv)

        self.results, self.param_grid = results, param_grid
