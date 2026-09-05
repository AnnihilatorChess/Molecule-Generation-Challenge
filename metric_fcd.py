import argparse
import os
import shutil
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)
parser.add_argument("--target", type=str, default='evaluation.tar')

args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmpdir:
    shutil.unpack_archive(args.target, tmpdir)
    print(f'Unpacked evaluation code/data to {tmpdir}')

    sys.path.insert(0, tmpdir)  # prepend on package search path
    from evaluation.evaluate_submission import get_metric  # import unpacked library

    args.trainset = os.path.join(tmpdir, 'evaluation/data/smiles_train.txt')
    args.teststats = os.path.join(tmpdir, 'evaluation/data/test_stats.p')

    metric_name = 'fcd'
    metric_value = get_metric(args, metric_name)
    print('#################################################################################################################################################')
    print(f'Metric name: {metric_name}')
    print(metric_value)
