#!/bin/bash
wandb_project='mixup-baseline-new'
#wandb_project='test'
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

#dataset='cifar100'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data
#epochs=300
#decay_1=100
#decay_2=200
#ngpu=1
#workers=2

#dataset='tiny-imagenet-200'
#labels_per_class=500
#data_dir=/h/ama/workspace/ama-at-vector/best-mix/data/tiny-imagenet-200
#epochs=1200
#decay_1=600
#decay_2=900
#ngpu=4
#workers=12

#arch='preactresnet18'
#arch='resnext29_4_24'
#arch='wrn16_8'

#prob_mix=1.0
lr=0.2

################cifar10/100
#for arch in 'preactresnet18' 'resnext29_4_24' 'wrn16_8';do
#for arch in 'resnext29_4_24';do
#for arch in 'wrn16_8';do
for arch in 'preactresnet18';do
	#vanilla
	#for seed in 0 1 2; do
		#j_name=$RANDOM
		#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train vanilla --ngpu ${ngpu} --workers ${workers}"
		#sleep 0.5
	#done

	#for prob_mix in 1.0 0.7; do
	#for prob_mix in 0.7; do
	for prob_mix in 1.0; do
		#input mixup
		#for seed in 0 1 2; do
			#j_name=$RANDOM
			#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --mixup_alpha 1.0 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
			#sleep 0.5
		#done

		#manifold mixup
		#for seed in 0; do
		for seed in 0 1 2; do
			j_name=$RANDOM
			bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup_hidden --mixup_alpha 2.0 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
			sleep 0.5
		done

		#cutmix
		#for seed in 0 1 2; do
			#j_name=$RANDOM
			#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --box True --mixup_alpha 1.0 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
			#sleep 0.5
		#done

		#puzzle
		#for seed in 0 1 2; do
			#j_name=$RANDOM
			#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --graph True --mixup_alpha 1.0 --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
			#sleep 0.5
		#done
	done
done


################tiny imagenet
#vanilla
#for seed in 0 0; do
	#j_name=$RANDOM
	#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train vanilla --ngpu ${ngpu} --workers ${workers}"
	#sleep 0.5
#done

#for prob_mix in 1.0 0.7; do
	##input mixup
	#for seed in 0 0; do
	##for seed in 0; do
		#j_name=$RANDOM
		#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --mixup_alpha 0.2 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
		#sleep 0.5
	#done

	##manifold mixup
	#for seed in 0 0; do
		#j_name=$RANDOM
		#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup_hidden --mixup_alpha 0.2 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
		#sleep 0.5
	#done

	##cutmix
	#for seed in 0 0; do
		#j_name=$RANDOM
		#bash launch_slurm_job.sh ${gpu} ${j_name} ${ngpu} "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --box True --mixup_alpha 0.2 --ngpu ${ngpu} --workers ${workers} --prob_mix ${prob_mix}"
		#sleep 0.5
	#done

	##puzzle
	#for seed in 0 0; do
		#j_name=$RANDOM
		#bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 main.py --dataset ${dataset} --data_dir ${data_dir} --labels_per_class ${labels_per_class} --arch ${arch}  --learning_rate ${lr} --momentum 0.9 --decay 0.0001  --epochs ${epochs} --schedule ${decay_1} ${decay_2} --gammas 0.1 0.1 --seed ${seed} --job_name ${j_name} --wandb_project ${wandb_project} --enable_wandb 1 --train mixup --graph True --mixup_alpha 1.0 --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_eps 0.8 --clean_lam 1 --ngpu 1 --workers 4 --prob_mix ${prob_mix}"
		#sleep 0.5
	#done
#done


