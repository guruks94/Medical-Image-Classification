#!/bin/bash
#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24000
#SBATCH -p gpu
#SBATCH -q wildfire

#SBATCH --gres=gpu:V100:1

#SBATCH -t 1-00:00:00
#SBATCH -o ./results/NIH-ChestX-Ray14/slurm.%j.out
#SBATCH -e ./results/NIH-ChestX-Ray14/slurm.%j.err
#SBATCH --mail-type=ALL                # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=gkempego@asu.edu   # send-to address

module purge
module load anaconda/py3
conda env list
source activate pyenv3
module load cuda/10.2.89
module load cudnn/8.1.0

cd /home/gkempego/Medical-Image-Classification
#python main_classification.py --resume False --exp_no 0 --num_class 14 --data_set ChestXray14 --init Random --data_dir /data/jliang12/mhossei2/Dataset/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt

#python main_classification.py --resume False --exp_no 0 --num_class 14 --data_set ChestXray14 --init ImageNet --data_dir /data/jliang12/mhossei2/Dataset/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt