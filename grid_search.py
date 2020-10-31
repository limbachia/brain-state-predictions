from src.preprocess.dataset import *
from src.models.model_selection import classifier, run_majority_vote, MyGridSearchCV
import os
import tensorflow as tf
import argparse

PROJ='/home/climbach/approach-retreat'
PROCPATH=os.path.join(PROJ,'data/processed')

# Common parameters to run the search grid on

def run(args):
    
    # Load data
    dataset = Dataset(args.input_data)
    dataset.load()
    print('Number of subjects: ',len(dataset.sid()))
    print('\n')
    print('Subject IDs:\n', dataset.sid())
    print('\n')
    print('Single subject data shape (timepoints, ROIs, trials*6):', 
          dataset.data['CON031']['data'].shape)
    
    '''
    Single subject data shape: (time x ROI x trials x shifts)
    Number of time points and ROIs should be consistent across subjects.
    Number of trials can vary across subjects.
    '''
    
    # Re-organize data
    '''
    Every trial consists of 14 timepoints. First 7 timepoints reflect 
    approach__ period and later 7 reflect __retreat__ period.
    Trials are split accordingly and given appropriate labels (approach = 0, retreat = 1). 
    '''
    if args.time_point == 'all':
        print('Training using all timepoints')
        dataset_df = organize_dataset(selective_segments(dataset.data))
    else:
        print('Training using the %ith timepoint'%int(args.time_point))
        dataset_df = organize_dataset(selective_segments(dataset.data,int(args.time_point)))
        
    # Train-test split
    dataset.train_test_split_sid()    
    
    # Perform grid search
    if os.path.isfile(args.out_data):
        print('Grid search results already exist')
    else:
        os.makedirs(os.path.dirname(args.out_data),exist_ok=True)
        params = {'L2':[float(val) for val in args.L2.split(' ')],
                  'dropout':[float(val) for val in args.dropout.split(' ')],
                  'lr':[float(val) for val in args.learning_rate.split(' ')]}
        print(params)
        grid_search = MyGridSearchCV()
        grid_search.fit(dataset_df, dataset.train_idx, classifier, params,cv=5,n_models=None)
        results = grid_search.results
        param_grid = grid_search.param_grid
        
        # Save the results and parameter grid in as pickle file
        with open(args.out_data,"wb") as f:
            pkl.dump([results,param_grid],f)
        
        print('Saved search-grid results: '+args.out_data)

if __name__ == '__main__':
    
    '''
    This script can be used to carry out gridsearch along with cv on multiple
    set of parameters, or to just perform cross validation using a 
    single set of parameters.
    '''
    
    parser = argparse.ArgumentParser(description='Perform grid search')
    
    parser.add_argument('-i','--input-data',type=str,
                        default = PROCPATH+'/00b-ROI316_withShiftedSegments.pkl',
                        help='path/to/segments/pkl/dataset')
    
    parser.add_argument('-tp','--time-point',type=str,
                        default='all',help='which timepoint to use: 0,1,2,3,4,5 and all are valid options')
    
    parser.add_argument('-L2','--L2',
                        type=str,
                        default='0 0.001 0.003 0.01 0.03',
                        help='L2 (alpha) regularization values (as floats)')
    
    parser.add_argument('-dropout','--dropout',
                        type=str,
                        default='0 0.1 0.2 0.3 0.4',
                        help='dropout fractions (as floats)')
    
    parser.add_argument('-lr','--learning-rate',
                        type=str,
                        default='0.001 0.003 0.006',
                        help='learning rate (lambda) values (as floats)')
    
    parser.add_argument('-cv','--cv',type=int,default=5,help='number of folds for cross-validation')
    
    parser.add_argument('-n','--n-models',type=int,
                        default=None,
                        help='number of random models to run')
    
    parser.add_argument('-o','--out-data',type=str,
                        default=PROJ+'/results/00-ROI316_withShiftedSegments/grid_search_results.pkl',
                        help='number of random models to run')
    
    args = parser.parse_args()

    run(args)
