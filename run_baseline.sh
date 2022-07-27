#!/bin/bash
wandb_project='bmvc-run-time-comparison'
#wandb_project='test'
#gpu="t4v1,p100,t4v2,rtx6000"
gpu="t4v2"
#gpu="t4v1,t4v2"
#gpu="t4v1,p100,t4v2"
#gpu="p100"
#gpu="rtx6000"

#dataset='cifar10'
#labels_per_class=5000
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
#epochs=3
#decay_1=100
#decay_2=200
#ngpu=1
#workers=2

dataset='cifar100'
labels_per_class=500
data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
epochs=3
decay_1=100
decay_2=200
ngpu=1
workers=2

#dataset='tiny-imagenet-200'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data/tiny-imagenet-200
#epochs=3
#decay_1=600
#decay_2=900
##ngpu=4
##workers=12
#ngpu=1
#workers=2

#arch='preactresnet18'
#arch='resnext29_4_24'
#arch='wrn16_8'

prob_mix=1.0
lr=0.2

################cifar10/100
#for arch in 'preactresnet18' 'resnext29_4_24' 'wrn16_8';do
for arch in 'resnext29_4_24';do
#for arch in 'wrn16_8';do
#for arch in 'preactresnet18';do
	for seed in 0; do
		#for method in 'vanilla' 'input' 'manifold' 'cutmix' 'puzzle'; do
		for method in 'input'; do
			j_name=$RANDOM$RANDOM
			bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --method ${method} --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix} --new_implementation 0 --wandb_EOT 1"
			sleep 0.5
		done
	done
done
