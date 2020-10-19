#GPUS=32 ./tools/slurm_train.sh ad_lidar baseline configs/imagenet/hrnet.py result_hrnet
CUDA_VISIBLE_DEVICES=6,7 PORT=29500 ./tools/dist_train.sh configs/hrnet.py 2 --work-dir ./output/hrnet
CUDA_VISIBLE_DEVICES=0,1 PORT=29522 ./tools/dist_train.sh configs/res50.py 2 --work-dir ./output/res50_notrans
CUDA_VISIBLE_DEVICES=2,3 PORT=29523 ./tools/dist_train.sh configs/res18.py 2 --work-dir ./output/res18_notrans
CUDA_VISIBLE_DEVICES=2,3 PORT=29555 ./tools/dist_train.sh configs/shufflenet.py 2 --work-dir ./output/shufflenet
CUDA_VISIBLE_DEVICES=4,5 PORT=29556 ./tools/dist_train.sh configs/mobilenet.py 2 --work-dir ./output/mobilenet