# coding:utf-8

from glob import glob
import os

nums = 1000
src_norm_root = '/media/wf/Data2/20190222/TCGADI/train/KICH/25normal/'


tar_root = '/home/wf/code/data/Patches_TCGA/'
tar_norm_root = tar_root + "0/"
tar_cancer_root = tar_root + "1/"

norm_tiles = glob(src_norm_root + "*/299_299/*.jpeg")
BAG_COUNT = 15

mid = len(norm_tiles) // BAG_COUNT // 2

for i in range(mid):
    tar_dir = tar_norm_root + "img" + str(i)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    for src_tile in norm_tiles[BAG_COUNT * i: BAG_COUNT * (i + 1)]:
        prefix = src_tile.split('/')[-3][:12]
        tar_tile = os.path.join(tar_dir, prefix + '_' + os.path.basename(src_tile))
        os.symlink(src_tile, tar_tile)

for i in range(mid + 1, len(norm_tiles) // BAG_COUNT):
    tar_dir = tar_cancer_root + "img" + str(i)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    for src_tile in norm_tiles[BAG_COUNT * i: (BAG_COUNT * (i + 1)-1)]:
        prefix = src_tile.split('/')[-3][:12]
        tar_tile = os.path.join(tar_dir, prefix + '_' + os.path.basename(src_tile))
        os.symlink(src_tile, tar_tile)

# TODO：add cancer tile to cancer dir
src_cancer_root = '/media/wf/Data2/20190222/TCGADI/train/KICH/25cancer/'
cancer_tiles = glob(src_cancer_root + "*/299_299/*.jpeg")