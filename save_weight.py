import os
import pickle
import subprocess

import click
import tensorflow as tf

from baselines.her.util import save_weight


@click.command()
@click.option('--policy_file', type=str, default=None)
@click.option('--run_group', type=str, default=None)
@click.option('--epoch', type=int, default=None)
def main(policy_file, run_group, epoch):
    import glob
    tf.compat.v1.disable_eager_execution()

    if policy_file is not None:
        policy_file = glob.glob(policy_file)[0]
        base = os.path.splitext(policy_file)[0]
        with open(policy_file, 'rb') as f:
            pretrain = pickle.load(f)
        pretrain_weights = save_weight(pretrain.sess)
        output_file = open(base + '_weight.pkl', 'wb')
        pickle.dump(pretrain_weights, output_file)
        output_file.close()
    else:
        runs = glob.glob(f'logs/{run_group}*/*')
        print(runs)
        for run in sorted(runs):
            policy_file = f'{run}/policy_{epoch}.pkl'
            print(policy_file)
            subprocess.Popen(['python', 'save_weight.py', f'--policy_file={policy_file}'])

if __name__ == '__main__':
    main()
