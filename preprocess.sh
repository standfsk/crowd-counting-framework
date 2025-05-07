#!/bin/sh
# !!! ----------------- Please change the `src_dir` to the location of your dataset. ----------------- !!!
# !!! ----- Please do NOT change `dst_dir` as it will be used to locate the dataset in crowd.py. ----- !!!

python preprocess.py --dataset shanghaitech_a --src_dir ./datasets/ShanghaiTech/part_A --dst_dir ./datasets/sha  --min_size 512 --max_size 1920
python preprocess.py --dataset shanghaitech_b --src_dir ./datasets/ShanghaiTech/part_B --dst_dir ./datasets/shb  --min_size 512 --max_size 1920
#python preprocess.py --dataset nwpu           --src_dir ./datasets/NWPU-Crowd          --dst_dir ./datasets/nwpu --min_size 512 --max_size 1920
#python preprocess.py --dataset ucf_qnrf       --src_dir ./datasets/UCF-QNRF            --dst_dir ./datasets/qnrf --min_size 512 --max_size 1920
#python preprocess.py --dataset jhu            --src_dir ./datasets/jhu_crowd_v2.0      --dst_dir ./datasets/jhu  --min_size 512 --max_size 1920