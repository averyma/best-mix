#!/bin/bash
wandb_project='test'
#wandb_project='mixup-ours-sweep'
gpu="t4v1,p100,t4v2,rtx6000"
dataset='cifar10'
labels_per_class=5000
#dataset='cifar100'
#labels_per_class=500
arch='preactresnet18'
eval_mode=0
grad_normalization=0
kernel_size=5

#for seed in 0 1 2; do
for seed in 0; do
    for lr in 0.2; do
        for blur_sigma in 1.0; do
            for use_yp_argmax in 0; do
                j_name=$RANDOM

                bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 main.py --dataset cifar10 --data_dir /h/ama/workspace/ama-at-vector/PuzzleMix-master/data --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001 --epochs 300 --schedule 100 200 --gammas 0.1 0.1 --train ours  --use_yp_argmax ${use_yp_argmax} --blur_sigma ${blur_sigma} --eval_mode ${eval_mode} --grad_normalization ${grad_normalization} --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --kernel_size ${kernel_size}"
                sleep 0.5
            done
        done
    done
done
