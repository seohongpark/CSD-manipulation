# Controllability-Aware Unsupervised Skill Discovery


## Overview
This is the official implementation of [**Controllability-aware Skill Discovery** (**CSD**)](https://arxiv.org/abs/2302.05103) on manipulation environments (Fetch and Kitchen).
The codebase is based on the implementation of [MUSIC](https://github.com/ruizhaogit/music).
We refer to http://github.com/seohongpark/CSD-locomotion for the implementation of CSD on locomotion environments.

Please visit [our project page](https://seohong.me/projects/csd/) for videos.

## Installation

```
conda create --name csd-manipulation python=3.8
conda activate csd-manipulation
pip install -r requirements.txt
```


## Examples

FetchPush (2-D continuous skills)
```
python train.py --run_group Exp --env_name FetchPush-v1 --n_epochs 1002 --num_cpu 1 --logging True --note DIAYN --hidden 256 --layers 2 --skill_type continuous --num_skills 2 --n_cycles 40 --policy_save_interval 500 --plot_freq 25 --plot_repeats 4 --max_path_length 50 --n_batches 10 --rollout_batch_size 2 --sk_clip 0 --et_clip 1 --seed 0 --buffer_size 100000 --polyak 0.995 --algo_name csd --inner 1 --algo csd --dual_reg 1 --dual_lam_opt adam --dual_dist s2_from_s --dual_init_lambda 3000 --dual_slack 1e-06 --train_start_epoch 50 --sk_r_scale 500 --et_r_scale 0.02
```
FetchSlide (2-D continuous skills)
```
python train.py --run_group Exp --env_name FetchSlide-v1 --n_epochs 1002 --num_cpu 1 --logging True --note DIAYN --hidden 256 --layers 2 --skill_type continuous --num_skills 2 --n_cycles 40 --policy_save_interval 500 --plot_freq 25 --plot_repeats 4 --max_path_length 50 --n_batches 10 --rollout_batch_size 2 --sk_clip 0 --et_clip 1 --seed 0 --buffer_size 100000 --polyak 0.995 --algo_name csd --inner 1 --algo csd --dual_reg 1 --dual_lam_opt adam --dual_dist s2_from_s --dual_init_lambda 3000 --dual_slack 1e-06 --train_start_epoch 50 --sk_r_scale 500 --et_r_scale 0.02
```
FetchPickAndPlace (3-D continuous skills)
```
python train.py --run_group Exp --env_name FetchPickAndPlace-v1 --n_epochs 1002 --num_cpu 1 --logging True --note DIAYN --hidden 256 --layers 2 --skill_type continuous --num_skills 3 --n_cycles 40 --policy_save_interval 500 --plot_freq 25 --plot_repeats 4 --max_path_length 50 --n_batches 10 --rollout_batch_size 2 --sk_clip 0 --et_clip 1 --seed 0 --buffer_size 100000 --polyak 0.995 --algo_name csd --inner 1 --algo csd --dual_reg 1 --dual_lam_opt adam --dual_dist s2_from_s --dual_init_lambda 3000 --dual_slack 1e-06 --train_start_epoch 50 --sk_r_scale 500 --et_r_scale 0.02
```
Kitchen (2-D continuous skills)
```
python train.py --run_group Exp --env_name Kitchen --n_epochs 502 --num_cpu 1 --logging True --note DIAYN --hidden 256 --layers 2 --skill_type continuous --num_skills 2 --n_cycles 40 --policy_save_interval 500 --plot_freq 25 --plot_repeats 4 --max_path_length 50 --n_batches 10 --rollout_batch_size 2 --sk_clip 0 --et_clip 1 --seed 0 --buffer_size 100000 --polyak 0.995 --n_random_trajectories 50 --algo_name csd --inner 1 --algo csd --dual_reg 1 --dual_lam_opt adam --dual_dist s2_from_s --dual_init_lambda 3000 --dual_slack 1e-06 --train_start_epoch 50 --sk_r_scale 500 --et_r_scale 0.02
```
Kitchen (16 discrete skills)
```
python train.py --run_group Exp --env_name Kitchen --n_epochs 502 --num_cpu 1 --logging True --note DIAYN --hidden 256 --layers 2 --skill_type discrete --num_skills 16 --n_cycles 40 --policy_save_interval 500 --plot_freq 25 --plot_repeats 4 --max_path_length 50 --n_batches 10 --rollout_batch_size 2 --sk_clip 0 --et_clip 1 --seed 0 --buffer_size 100000 --polyak 0.995 --n_random_trajectories 50 --algo_name csd --inner 1 --algo csd --dual_reg 1 --dual_lam_opt adam --dual_dist s2_from_s --dual_init_lambda 3000 --dual_slack 1e-06 --train_start_epoch 50 --sk_r_scale 500 --et_r_scale 0.02
```

## Licence

MIT
