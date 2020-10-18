scp -P 9001 -r tiger@10.188.180.34:/opt/tiger/uslabcv/dingmingyu/imagenet-50-1000/ .

git clone https://github.com/dingmyu/mmclassification.git
cd mmclassification
mkdir data
cd data
ln -s /opt/tiger/uslabcv/dingmingyu/imagenet-50-1000 imagenet
cd -
python3 setup.py develop --user


rsync -azv * myding@10.200.64.72:/Volumes/Mingyu_Research/NAS_benchmark/cls/



ps aux | grep search | awk '{print $2}' | xargs kill -9
ps aux | grep hrnet | awk '{print $2}' | xargs kill -9