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
    for model in models:
        if not force and os.path.isfile(os.path.join(model, f'{model}.h5')):
            continue
        print(f'predicting {model}')
        os.system(f'python -m imet.main predict_test {model} --model {model.split("_")[1]}')


def make_submission(models):
    models = ' '.join(map(lambda pred: os.path.join(model_path, f'{model_path}.h5'), models))
    os.system(f'python -m imet.make_submission {models} submission.csv --threshold 0.1')


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('models', nargs='+')        
    arg('force', type=int)
    args = parser.parse_args()
    models = [f'model_{model}_fold_{fold}' for model in args.models for fold in range(5)]
    print(models)
    make_predicts(models, args.force)
    make_submission(models)


if __name__ == '__main__':
    main()
