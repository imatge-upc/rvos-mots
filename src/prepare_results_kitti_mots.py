import sys, os
import os.path as osp
import os
from args import get_parser
import numpy as np
from PIL import Image
from misc.config_kittimots import cfg
from utils.utils import make_dir
import json
import time

def get_dict(seq_name):

    dict_root_dir = cfg.PATH.ANNOTATIONS
    path = osp.join(dict_root_dir, seq_name, 'dictionary.txt')
    txt = open(path, 'r')
    dict = json.loads(txt.read())

    return dict


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--model_name', dest='model_name', default='model')
    parser.add_argument('--num_instances', dest='num_instances', default=90)
    args = parser.parse_args()

    submission_base_dir = osp.join('../models', args.model_name, 'Annotations-kitti/')
    make_dir(submission_base_dir)

    seq_base_dir = osp.join('../models', args.model_name, 'masks_sep_2assess-kitti/')
    sequences = os.listdir(seq_base_dir)

    for seq_name in sequences:
        submission_dir = osp.join(submission_base_dir, seq_name)
        make_dir(submission_dir)

        dict_seq = get_dict(seq_name)
        images = sorted(os.listdir(osp.join(seq_base_dir, seq_name)))

        shape = (256,448)
        kitti_mask = np.zeros(shape)
        for img in images:
            pred_mask = np.array(Image.open(osp.join(seq_base_dir, seq_name, img)))

            num_frame = int(img[:6])
            num_instance = int(img[16:18]) + 1

            indx = np.where(pred_mask == 255)
            if len(indx[0]) != 0:
                kitti_mask[indx] = dict_seq.get(str(num_instance))

            if num_instance == int(args.num_instances):
                res_im = Image.fromarray(kitti_mask, mode="P")
                res_im.save(submission_dir + '/%06d.png' % num_frame)
                kitti_mask = np.zeros(pred_mask.shape)


