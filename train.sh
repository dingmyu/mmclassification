#GPUS=32 ./tools/slurm_train.sh ad_lidar baseline configs/imagenet/hrnet.py result_hrnet
CUDA_VISIBLE_DEVICES=6,7 PORT=29500 ./tools/dist_train.sh configs/hrnet.py 2 --work-dir ./output/hrnet
CUDA_VISIBLE_DEVICES=6,7 PORT=29520 ./tools/dist_train.sh configs/res50.py 2 --work-dir ./output/res50
CUDA_VISIBLE_DEVICES=4,5 PORT=29521 ./tools/dist_train.sh configs/res18.py 2 --work-dir ./output/res18