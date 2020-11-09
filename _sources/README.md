# Brain State Predictions During Approaching and Retreating Threats: An fMRI Study.


Functional magnetic resonance imaging (fMRI) data, RNN implementation to predict brain states during anticipation of aversive and neutral events.

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

> CUDA version 10.0.130

Data: Proprietary

## Tuning, Training & Evaluation
To find best hyperparameters like `L2`, `dropout`, and `learning_rate` for the GRU classifier
using a cross-validation, gri-search approach:

```
python grid_search.py --input-data <path to segments dataset pickle file> \
    --time-point 5 --L2 '0 0.001 0.003 0.01 0.03' --dropout '0 0.1 0.2 0.3 0.4' \
    --learning-rate '0.001 0.003 0.006' --cv <number of folds> \
    --n-models <number of random models to run a randomized grid-search (skip to search the entire grid)> \
    --out-data <declare path and output file name (end with extension '.pkl')>
```

The `grid_search.py` script can be used to find the best combination of `L2`, `dropout`, and `learning_rate` by specifying a range of values for each hyperparameter (as in the above example). It can also be used to simply get a k-fold cross-validation performance for a single combination of hyperparameters. This can be done by specifying single values for every hyperparameter and skipping the `--n-models` option.

---

To train a model using the best hyperparameters and save it:
```
python train_model.py -GSCV <path to the output of grid_search.py> \
    -data <path to segments dataset pickle file> \
    -tp 5 -overwrite <overwrite previous output: 0 (default) or 1> \
    -o <declare path and model name (end with extension '.h5')>
```

---
To get chance chance accuracy distribution:
```
python perm_accuracy.py -data <path to segments dataset pickle file> \
    -tp 5 --best-L2 <float> --best-dropout <float> --best-lerning-rate <float> \
    -k_perms <number of permutations to perform (default is 1000)> \
    -overwrite <overwrite previous output: 0 (default) or 1> \
    --output <declare path and output file name (end with extension '.pkl')>
```


