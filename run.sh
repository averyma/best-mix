#!/bin/bash
#wandb_project='bmvc-run-time-comparison'
#wandb_project='bmvc-cifar10-100-shift-comparison'
#wandb_project='bmvc-cifar10-100-higher-density'
#wandb_project='bmvc-cifar-eval-train'
#wandb_project='bmvc-cifar100-ablation'
#wandb_project='bmvc-cifar100-ablation-resnext'
#wandb_project='bmvc-cifar100-ablation-resnext'
#wandb_project='ours-tiny-eps-fix'
wandb_project='bmvc-cifar10-100-SM-ablation'
#wandb_project='ours-v4-v5-comparison'
#wandb_project='ours-with-shift-cifar100-resnext-fixed-sampling-update-ratio'
#wandb_project='ours-with-shift-cifar10-alpha1-1-rand-pos'
#wandb_project='ours-with-shift-cifar100-resnext'
#wandb_project='ours-with-shift-tiny-fixed-sampling-update-ratio'
#wandb_project='mixup-baseline'
#wandb_project='ours-best-old-setup'
#wandb_project='ours-cifar100-resnext29-4-24'
#wandb_project='ours-n'
#wandb_project='ours-cifar100-wrn16-8'
#wandb_project='ours-cifar100-preactresnet18'
#wandb_project='ours-cifar100-preactresnet18-new'
#wandb_project='ours-cifar10-preactresnet18-new'
#gpu="rtx6000"
#gpu="t4v1,p100,t4v2,rtx6000"
#gpu="t4v1,t4v2"
gpu="t4v2"
#gpu="p100"
#gpu="rtx6000"
#gpu="p100,rtx6000"
wandb_mode="online"
wandb_EOT=1
wandb_log_freq=10
date=`date +%Y%m%d`

#dataset='cifar10'
#labels_per_class=5000
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
#epochs=300
#decay_1=100
#decay_2=200
#ngpu=1
#workers=2
#kernel_size=5

dataset='cifar100'
labels_per_class=500
data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
epochs=300
decay_1=100
decay_2=200
ngpu=1
workers=2
kernel_size=5

#dataset='tiny-imagenet-200'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data/tiny-imagenet-200
#epochs=1200
#decay_1=600
#decay_2=900
#decay_3=1000
##ngpu=
##workers=12
#ngpu=1
#workers=2
#kernel_size=11

arch='preactresnet18'
#arch='resnext29_4_24'
#arch='wrn16_8'

lr=0.2
new_imp=1

eval_mode=0

mixup_alpha=1.0
bce_saliency=1
grad_normalization='L1'
with_shift=1
no_SM=1

#for seed in 0; do
for seed in 2; do
	#for blur_sigma in 2.0 100; do
	#for blur_sigma in 1.0 2.0; do
	for blur_sigma in 1.0; do
	#for blur_sigma in 2.0; do
	#for blur_sigma in 2.0 3.0; do
	#for blur_sigma in 100.0; do
		#for upper_lambda in 0.7; do
		#for upper_lambda in 0.5 0.6 0.7; do
		#for upper_lambda in 0.5 0.6 0.7 1.0; do
		#for upper_lambda in 0.6; do
		for upper_lambda in 0.5; do
		#for upper_lambda in 0.6 0.7; do
		#for upper_lambda in 0.5 0.6; do
			#for rand_pos in 1 0; do
			#for rand_pos in 0.1 0.7 0.9; do
			#for rand_pos in 0.01 0.025 0.05 0.075 0.1; do
			#for rand_pos in 0.065; do
			#for rand_pos in 0.01 0.05 0.1; do
			for rand_pos in 0.01; do
			#for rand_pos in 0.5 0.7 1.0; do
			#for rand_pos in 0.001; do
			#for rand_pos in 0.1; do
				#for update_ratio in 0.3 0.5 0.7; do
				#for update_ratio in 1.0; do
				for update_ratio in 0.7; do
				#for update_ratio in 1.0 0.7 0.5; do
					j_name=${date}$RANDOM$RANDOM
					bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001 --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --method ours --blur_sigma ${blur_sigma} --eval_mode ${eval_mode} --grad_normalization ${grad_normalization} --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --wandb_mode ${wandb_mode} --kernel_size ${kernel_size} --ngpu ${ngpu} --workers ${workers} --new_implementation ${new_imp} --mixup_alpha ${mixup_alpha} --upper_lambda ${upper_lambda} --rand_pos ${rand_pos} --bce_saliency ${bce_saliency} --wandb_log_freq ${wandb_log_freq} --wandb_EOT  ${wandb_EOT} --update_ratio ${update_ratio} --with_shift ${with_shift} --no_SM ${no_SM}"
					#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001 --epochs ${epochs} --schedule ${decay_1} ${decay_2} ${decay_3} --gammas 0.1 0.1 0.1 --method ours --blur_sigma ${blur_sigma} --eval_mode ${eval_mode} --grad_normalization ${grad_normalization} --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --wandb_mode ${wandb_mode} --kernel_size ${kernel_size} --ngpu ${ngpu} --workers ${workers} --new_implementation ${new_imp} --mixup_alpha ${mixup_alpha} --upper_lambda ${upper_lambda} --rand_pos ${rand_pos} --bce_saliency ${bce_saliency} --wandb_log_freq ${wandb_log_freq} --wandb_EOT  ${wandb_EOT} --update_ratio ${update_ratio} --with_shift 1"
					sleep 0.5
				done
			done
		done
	done
done
