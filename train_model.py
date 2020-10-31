import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os

from src.preprocess.dataset import *
from src.models.model_selection import classifier

def _info(s):
    print('---------')
    print(s)
    print('---------')

def run(args):
    print(args.GSCV)
    print(args.data)
    print(bool(args.overwrite))
    print(args.output)
    
    if not bool(args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(args.output+' already exists.\nTo overwite, use option "-overwrite 1"')
        
        
    with open(args.GSCV,"rb") as f:
        results, param_grid = pickle.load(f)
        
    table = pd.DataFrame.from_dict({(i,j,k): results[i][j][k] for i in results.keys() for j in results[i].keys() for k in results[i][j].keys()}).T
    table.reset_index(inplace=True)
    table.rename(columns={'level_0':'model','level_1':'fold','level_2':'set',0:'acc'},inplace=True)
    # Find the best model; model that has highest val mean (acorss folds) accuracy and lowest standard deviation (acorss folds).
    model_num = table[table['set']=='val'].groupby(['model'])['acc'].agg([np.mean,np.std])
    model_num.reset_index(inplace=True)
    model_num = model_num.sort_values(by=['mean','std'],ascending=[False,True])
    model_num.reset_index(drop=True,inplace=True)
    model_num = int(model_num['model'][0].replace('model',''))
    
    print('\nBest parameters: ')
    print(param_grid[model_num])
    
    # load data
    print('\nLoading '+args.data+'...')
    dataset = Dataset(args.data)
    dataset.load()
    print('\nTraining using the %ith timepoint'%int(args.time_point))
    dataset_df = organize_dataset(selective_segments(dataset.data,int(args.time_point)))
    dataset.train_test_split_sid()
    
    X_train, y_train = query_dataset(dataset_df,dataset.train_idx)
    X_test, y_test = query_dataset(dataset_df,dataset.test_idx)
    print('\nTraining on %i/%i participants' %(len(dataset.train_idx),len(dataset.sid())))
    
    model = classifier(X_train,y_train, param_grid[model_num]['L2'],
                       param_grid[model_num]['dropout'],param_grid[model_num]['lr'])
    
    train_loss, train_acc = model.evaluate(X_train,y_train)
    print('\nTraining Accuracy: %.2f'%train_acc)
    test_loss, test_acc = model.evaluate(X_test,y_test)
    print('\nTest Accuracy: %.2f'%test_acc)

    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    #model.save(args.output)
    print('\nmodel.save({})'.format(args.output))
    
if __name__ == "__main__":
    '''
    This script should run after results from running grid_search.py 
    have been saved at results/*/grid_earch.pkl
    '''
    parser = argparse.ArgumentParser(description='Extracts best parameters from the grid_search.py results, tarins and svaes the model')
    
    parser.add_argument('-GSCV',type=str,help='grid_search/results/pickle/file')
    parser.add_argument('-data',type=str,help='path/to/data/pickle/file')
    parser.add_argument('-tp','--time-point',type=str,
                        default='5',help='which timepoint to use: 0,1,2,3,4,5 and all are valid options')
    parser.add_argument('-overwrite',type=int,default=0,help='overwrite existing model = 0 or 1 (default is 0)')
    parser.add_argument('-o','--output',type=str,help='path/to/save/model')
    
    args = parser.parse_args()
    run(args)
    