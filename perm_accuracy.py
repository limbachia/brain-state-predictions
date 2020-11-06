from src.preprocess.dataset import *
from src.models.model_selection import classifier, permute
import tensorflow as tf
import os
import numpy as np
import argparse
import pickle

def run(args):
    
    if not bool(args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(args.output+' already exists.\nTo overwite, use option "-overwrite 1"')
        
    # Load data
    dataset = Dataset(args.data)
    dataset.load()
    if args.time_point == 'all':
        print('Training using all timepoints')
        dataset_df = organize_dataset(selective_segments(dataset.data))
    elif args.time_point == 'None':
        dataset_df = organize_dataset(selective_segments(dataset.data,None))
    else:
        print('Training using the %ith timepoint'%int(args.time_point))
        dataset_df = organize_dataset(selective_segments(dataset.data,int(args.time_point)))
    dataset.train_test_split_sid()
    
    X_train, y_train = query_dataset(dataset_df,dataset.train_idx)
    X_test, y_test = query_dataset(dataset_df,dataset.test_idx)
    
    print('\nBest Parameters:')
    params = {'L2':args.best_L2,
              'dropout':args.best_dropout,
              'lr':args.best_learning_rate}
    print(params)
    
    # Load the model
    model = classifier(X_train, y_train, 
                       params['L2'],
                       params['dropout'],
                       params['lr'])
    
    _, obs_acc = model.evaluate(X_test,y_test)
    
    results = permute(X_train, y_train, X_test, y_test, 
                      params, k_perm=args.k_perms)
    
    
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"wb") as f:
        pickle.dump([dict(obs_test_acc=obs_acc),results],f)
    
if __name__ == '__main__':
    '''
    This script chance accuracy distribution
    '''
    
    parser = argparse.ArgumentParser(description="A classifier with best hyperparameters is trained on the training set with shuffled lables k_perm number of times. After every training iteration, the classifier if tested on a validation set with non-shuffled labels. This process is meant to simulate a chance accuracy distribution. The mean of the chance accuracy distribution can be used as a baseline performance measure against the observed performance of the of the classifer when trained on non-shuffled training set.")
    
    parser.add_argument('-data',type=str,help='path/to/data')
    parser.add_argument('-tp','--time-point',
                        type=str,
                        default='5',
                        help='which timepoint to use: 0,1,2,3,4,5 and all are valid options (default = 5).')
    parser.add_argument('--best-L2',type=float,default=0.003,help='best L2')
    parser.add_argument('--best-dropout',type=float,default=0.3,help='best dropout rate')
    parser.add_argument('--best-learning-rate',type=float,default=0.001,help='best learning_rate')
    parser.add_argument('-k_perms',type=int,default=1000,help='number of permutations (shuffles)')
    parser.add_argument('-overwrite',type=int,default=0,help='overwrite existing output = 0 or 1 (default is 0)')
    parser.add_argument('-o','--output',type=str,help='output/path')
    args = parser.parse_args()
    run(args)
    
    