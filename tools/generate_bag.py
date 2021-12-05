# coding:utf-8
from glob import glob
import random
import os
import time

mag = '5.0'
src_dir = '/media/wf/移动盘3/中大前列腺癌数据/tile/0'
target_dir = '/media/wf/移动盘3/中大前列腺癌数据/train_5/attention/0'

# 一个目录升的bag数
bag_num = 5
bag_pic_num = 50
bag_pic_rate = 0.8

if __name__ == '__main__':
    t = int(round(time.time() * 1000))
    print(t)
    for pid_dir in glob(src_dir + '/*'):
        all_pics = glob(pid_dir + '/' + mag + '/*')
        sample_num = min(bag_pic_num, int(len(all_pics) * bag_pic_rate))
        print(f'process {pid_dir}, sample_num={sample_num}')
        for suffix in range(0, bag_num):
            tar_dir = target_dir + '/' + '_'.join([os.path.basename(pid_dir), str(t), str(suffix)])
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            for pic_selected in random.sample(all_pics, sample_num):
                os.symlink(pic_selected, tar_dir + '/' + os.path.basename(pic_selected))
