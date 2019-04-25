import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


model = 'resnet50'
fold = 0


run('python setup.py develop --install-dir /kaggle/working')
run('python -m imet.make_folds')
run(f'python -m imet.main train model_{model}_fold_{fold} --n-epochs 25 --model {model} --fold {fold}')
run(f'python -m imet.main predict_test model_{model}_fold_{fold} --model {model}')
run(f'python -m imet.make_submission model_{model}_fold_{fold}/test.h5 submission.csv --threshold 0.1')
