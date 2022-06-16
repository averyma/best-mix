#!/bin/bash
wandb_project='ours-with-shift-v2'
#wandb_project='mixup-baseline'
#wandb_project='ours-best-old-setup'
#wandb_project='ours-cifar100-resnext29-4-24'
#wandb_project='ours-n'
#wandb_project='ours-cifar100-wrn16-8'
#wandb_project='ours-cifar100-preactresnet18'
#wandb_project='ours-cifar100-preactresnet18-new'
#wandb_project='ours-cifar10-preactresnet18-new'
gpu="t4v1,p100,t4v2,rtx6000"
#gpu="t4v1,p100,t4v2"

dataset='cifar10'
labels_per_class=5000
data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
epochs=300
decay_1=100
decay_2=200
ngpu=1
workers=2
kernel_size=5

#dataset='cifar100'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
#epochs=300
#decay_1=100
#decay_2=200
#ngpu=1
#workers=2
#kernel_size=5

#dataset='tiny-imagenet-200'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data/tiny-imagenet-200
#epochs=1200
#decay_1=600
#decay_2=900
#ngpu=4
#workers=12
#kernel_size=11

arch='preactresnet18'
#arch='resnext29_4_24'
#arch='wrn16_8'

#mix_schedule='fixed'
mix_schedule='scheduled'
mix_scheduled_epoch=50
grad_normalization=0
lr=0.2
#new_imp=1

#prob_mix=0.7
#blur_sigma=1.75
#eval_mode=0
use_yp_argmax=0

#with_shift=1
mix_stride=1

for prob_mix in 1.0 0.7; do
	for new_imp in 1 0; do
		for eval_mode in 0 1; do
			for with_shift in 1 0; do
				for seed in 0 1 2 ; do
					for blur_sigma in 1.0 1.5 2.0 2.5 100; do
						j_name=$RANDOM$RANDOM
						bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001 --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --method ours  --use_yp_argmax ${use_yp_argmax} --blur_sigma ${blur_sigma} --eval_mode ${eval_mode} --grad_normalization ${grad_normalization} --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --kernel_size ${kernel_size} --prob_mix ${prob_mix} --mix_schedule ${mix_schedule} --mix_scheduled_epoch ${mix_scheduled_epoch} --ngpu ${ngpu} --workers ${workers} --new_implementation ${new_imp} --with_shift ${with_shift} --mix_stride ${mix_stride}"
						sleep 0.5
					done
				done
			done
		done
	done
done
