#!/usr/bin/bash
echo "Running command 1"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/cyclical_1 --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 1 --num_epoch 50
echo "Running command 3"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/mono_hw --kl_anneal_type Monotonic --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 1 --num_epoch 50
echo "Running command 4"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/none_hw --kl_anneal_type None --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 1 --num_epoch 50
echo "Running command 5"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/cyclical_hw_group --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 1 --num_epoch 50 --norm Group
echo "Running command 6"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/cyclical_instance --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 1 --num_epoch 50 --norm Instance
echo "Running command 7"
CUDA_VISIBLE_DEVICES=1 python Lab4_template/Trainer.py --DR /home/wujh1123/DLP/lab04/LAB4_Dataset/LAB4_Dataset --save_root /home/wujh1123/DLP/lab04/ckpt/cyclical_05 --fast_train --fast_train_epoch 10 --kl_anneal_ratio 1.5 --tfr 0.5 --num_epoch 50