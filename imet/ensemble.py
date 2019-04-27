import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm

from . import models
from .dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
from .transforms import train_transform, test_transform
from .utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE, loss_function, set_models_path_env)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['validate', 'predict'])
    arg('--model', default='resnet50')
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 6)
    arg('--tta', type=int, default=8)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--loss', type=str, default='')
    arg('--limit', type=int)
    arg('--input-size', type=int, default=288)
    args = parser.parse_args()

    run_roots = [Path(f'model_{args.model}_fold_{fold}') for fold in range(5)]
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train')
    set_models_path_env(args.model)
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    # train_fold = folds[folds['fold'] != args.fold]
    # valid_fold = folds[folds['fold'] == args.fold]
    
    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    criterion = loss_function(args.loss)
    model = getattr(models, args.model)(
        num_classes=N_CLASSES, pretrained=args.pretrained)
    use_cuda = cuda.is_available()

    if use_cuda:
        model = model.cuda()

    if args.mode == 'validate':
        if args.limit:
            folds = folds[:args.limit]
        valid_loader = make_loader(folds, test_transform)
        validation(model, run_roots, criterion, valid_loader,
            use_cuda=use_cuda, model_name=args.model)	

    elif args.mode.startswith('predict'):
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            use_cuda=use_cuda,
            workers=args.workers,
        )
        
        test_root = DATA_ROOT / (
            'test_sample' if args.use_sample else 'test')
        ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
        if args.use_sample:
            ss = ss[ss['id'].isin(set(get_ids(test_root)))]
        predict(model, run_roots, df=ss, root=test_root,
                out_path='test.h5',
                **predict_kwargs)


def predict(model, run_roots: list, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, use_cuda: bool):
    loader = DataLoader(
        dataset=TTADataset(root, df, test_transform, tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    for run_root in run_roots:
        print(run_root)
        load_model(model, run_root / 'best-model.pt')
        model.eval()
        with torch.no_grad():
            for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                all_outputs.append(outputs.data.cpu().numpy())
                all_ids.extend(ids)       
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def validation(
        model: nn.Module, run_roots, criterion, valid_loader, use_cuda, model_name
        ) -> Dict[str, float]:
    
    all_losses, all_predictions, all_targets = [], [], []
    for run_root in run_roots:
        print(run_root)
        load_model(model, run_root / 'best-model.pt')
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(valid_loader, desc='Validation'):
                all_targets.append(targets.numpy().copy())
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                all_losses.append(_reduce_loss(loss).item())
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


if __name__ == '__main__':
    main()
