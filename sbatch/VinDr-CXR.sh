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
#SBATCH -o ../results/VinDr-CXR/slurm.%j.out
#SBATCH -e ../results/VinDr-CXR/slurm.%j.err
#SBATCH --mail-type=ALL                # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=gkempego@asu.edu   # send-to address

module purge
module load anaconda/py3
conda env list
source activate pyenv-mic
module load cuda/10.2.89
module load cudnn/8.1.0

cd /home/gkempego/Medical-Image-Classification
#python main_classification.py --model Resnet50 --exp_no 0 --resume False --mode train --num_class 6 --init Random --data_set VinDrCXR --data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0 --train_list dataset/VinDrCXR_train_dicom.txt --val_list dataset/VinDrCXR_val_dicom.txt --test_list dataset/VinDrCXR_test_dicom.txt

#python main_classification.py --model Resnet50 --exp_no 0 --resume False --mode train --num_class 6 --init ImageNet --data_set VinDrCXR --data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0 --train_list dataset/VinDrCXR_train_dicom.txt --val_list dataset/VinDrCXR_val_dicom.txt --test_list dataset/VinDrCXR_test_dicom.txt


#python main_classification.py --model Efficientnet_B7 --exp_no 0 --resume False --mode train --num_class 6 --init Random --data_set VinDrCXR  --data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0 --train_list dataset/VinDrCXR_train_dicom.txt --val_list dataset/VinDrCXR_val_dicom.txt --test_list dataset/VinDrCXR_test_dicom.txt

#python main_classification.py --model Efficientnet_B7 --exp_no 0 --resume False --mode train --num_class 6 --init ImageNet --data_set VinDrCXR  --data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0 --train_list dataset/VinDrCXR_train_dicom.txt --val_list dataset/VinDrCXR_val_dicom.txt --test_list dataset/VinDrCXR_test_dicom.txt