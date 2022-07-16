#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name ArT --net deformable_resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 30 --resume model/pretrain/MLT/TextBPN_deformable_resnet50_300.pth --viz --viz_freq 300


#--viz --viz_freq 80
#--start_epoch 300
#--load_memory True
