import argparse
from itertools import islice
import json
from pathlib import Path
import warnings
from typing import Dict
import os

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning


def make_predicts(models, force=False):
    for model_path in models:
        print(model_path)
        model = "_".join(model_path.split("_")[1:-2])
        if not force and os.path.isfile(os.path.join(model_path, f'{model_path}.h5')):
            continue
        print(f'predicting {model_path} {model}')
        os.system(f'python -m imet.main predict_test {model_path} --model {model}')


def make_valid_predicts(models, force=False):
    for model in models:
        for fold in range(5):
            print(model, fold)
            model_path = f'model_{model}_fold_{fold}'
            if not force and os.path.isfile(os.path.join(model_path, f'val_{fold}.h5')):
                continue
            print(f'predicting {model_path} {model}')
            os.system(f'python -m imet.main predict_valid {model_path} --model {model} --fold {fold}')        


def make_submission(models):
    models = ' '.join(map(lambda model_path: os.path.join(model_path, f'{model_path}.h5'), models))
    os.system(f'python -m imet.make_submission {models} submission.csv --threshold 0.1')


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('models', nargs='+')        
    args = parser.parse_args()
    make_valid_predicts(args.models)


if __name__ == '__main__':
    main()
