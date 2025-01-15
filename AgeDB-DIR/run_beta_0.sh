CUDA_VISIBLE_DEVICES=0 python train.py --beta 0 --batch_size 256
CUDA_VISIBLE_DEVICES=1 python train.py --beta 0.5 --batch_size 256
CUDA_VISIBLE_DEVICES=2 python train.py --beta 1 --batch_size 256
CUDA_VISIBLE_DEVICES=2 python train.py --MSE --batch_size 256