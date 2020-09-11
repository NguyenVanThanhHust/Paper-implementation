import os
import os.path as osp
import argparse
import shutil
from distutils.dir_util import copy_tree

def copy_video(fp="../../ILSVRC2015/"):
    video_folder = osp.join(fp, "Data", "VID")
    val_folder = osp.join(video_folder, "val")
    dst_folder = osp.join(video_folder, "train", "val")
    copy_tree(val_folder, dst_folder)

def copy_anno(fp="../../ILSVRC2015/"):
    anno_folder = osp.join(fp, "Annotations", "VID")
    val_folder = osp.join(anno_folder, "val")
    dst_folder = osp.join(anno_folder, "train", "val")
    copy_tree(val_folder, dst_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', default="../../ILSVRC2015/", type=str)
    args = parser.parse_args()
    copy_video(args.src_folder)  
    copy_anno(args.src_folder)  
