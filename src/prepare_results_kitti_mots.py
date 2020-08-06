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
from torchvision import transforms


def get_dict(seq_name):

    dict_root_dir = cfg.PATH.ANNOTATIONS
    path = osp.join(dict_root_dir, seq_name, 'dictionary.txt')
    txt = open(path, 'r')
    dict = json.loads(txt.read())

    return dict


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--model_name', dest='model_name', default='model')
    #parser.add_argument('--num_instances', dest='num_instances', default=90)
    args = parser.parse_args()

    submission_base_dir = osp.join('../models', args.model_name, 'Annotations-kitti/')
    make_dir(submission_base_dir)

    seq_base_dir = osp.join('../models', args.model_name, 'masks_sep_2assess-kitti/')
    sequences = sorted(os.listdir(seq_base_dir))

    for seq_name in sequences:
        submission_dir = osp.join(submission_base_dir, seq_name)
        make_dir(submission_dir)

        dict_seq = get_dict(seq_name)
        images = sorted(os.listdir(osp.join(seq_base_dir, seq_name)))

        shape = (287, 950)
        if seq_name == "0014" or seq_name == "0016":
            final_shape = (370,1224)
        elif seq_name == "0018":
            final_shape = (374, 1238)
        else:
            final_shape = (375, 1242)
        kitti_mask = np.zeros(shape, dtype=int)
        first_img = images[0]
        num_first_frame = int(first_img[:6])
        frame = 0
        while num_first_frame != frame:
            zeros = np.zeros(final_shape)
            im = Image.fromarray(zeros.astype(np.uint32))
            im.save(submission_dir + '/%06d.png' % frame)
            frame += 1

        old_num_frame = num_first_frame
        #old_num_frame = 328

        for img in images:


            pred_mask = np.array(Image.open(osp.join(seq_base_dir, seq_name, img)))

            num_frame = int(img[:6])
            num_instance = int(img[16:18])

            instance_dict = {}

            if num_frame != old_num_frame:

                res_im = Image.fromarray(kitti_mask.astype(np.uint32))
                resize = transforms.Resize(final_shape, interpolation=Image.NEAREST)
                res_im = resize(res_im)

                res_im.save(submission_dir + '/%06d.png' % (old_num_frame))

                kitti_mask = np.zeros(pred_mask.shape)


                while num_frame > (old_num_frame+1):
                    zeros = np.zeros(final_shape)
                    im = Image.fromarray(zeros.astype(np.uint32))
                    im.save(submission_dir + '/%06d.png' % (old_num_frame +1))
                    old_num_frame = old_num_frame + 1

            indx = np.where(pred_mask == 255)
            for (x,y) in zip(indx[0], indx[1]):
                if kitti_mask[x,y] != 0:

                    overlapping_id = kitti_mask[x,y]

                    if str(overlapping_id) in instance_dict:
                        kitti_mask[x, y] = instance_dict.get(str(overlapping_id))

                    else:
                        overlapping_id = int(list(dict_seq.keys())[list(dict_seq.values()).index(overlapping_id)])

                        path_t1_prev = osp.join(seq_base_dir, seq_name , '%06d' % (num_frame-2) + "_instance_" + "%02d.png" % num_instance)
                        if osp.exists(path_t1_prev):
                            t1_prev_frame = np.array(Image.open(path_t1_prev))
                            t1_prev_frame_idx = np.where(t1_prev_frame == 255)
                        else:
                            t1_prev_frame_idx = [[],[]]

                        path_t1_prev_overlapping = osp.join(seq_base_dir, seq_name , '%06d' % (num_frame-2) + "_instance_" + "%02d.png" % overlapping_id)
                        if osp.exists(path_t1_prev_overlapping):
                            t1_prev_frame_overlapping = np.array(Image.open(path_t1_prev_overlapping))
                            t1_prev_frame_idx_overlapping = np.where(t1_prev_frame_overlapping == 255)
                        else:
                            t1_prev_frame_idx_overlapping = [[],[]]

                        path_t2_prev = osp.join(seq_base_dir, seq_name, '%06d' % (num_frame - 3) + "_instance_" + "%02d.png" % num_instance)
                        if osp.exists(path_t2_prev):
                            t2_prev_frame = np.array(Image.open(path_t2_prev))
                            t2_prev_frame_idx = np.where(t2_prev_frame == 255)
                        else:
                            t2_prev_frame_idx = [[],[]]

                        path_t2_prev_overlapping = osp.join(seq_base_dir, seq_name, '%06d' % (num_frame - 3) + "_instance_" + "%02d.png" % overlapping_id)
                        if osp.exists(path_t2_prev_overlapping):
                            t2_prev_frame_overlapping = np.array(Image.open(path_t2_prev_overlapping))
                            t2_prev_frame_idx_overlapping = np.where(t2_prev_frame_overlapping == 255)
                        else:
                            t2_prev_frame_idx_overlapping = [[],[]]

                        area_idx =abs(len(t1_prev_frame_idx[0]) - len(t2_prev_frame_idx[0]))
                        area_idx_overlapping = abs(len(t1_prev_frame_idx_overlapping[0]) - len(t2_prev_frame_idx_overlapping[0]))

                        value = dict_seq.get(str(overlapping_id))
                        if area_idx > area_idx_overlapping:
                            instance_dict.update({value:int(value)})
                            continue
                        else:
                            if len(indx[0]) != 0:
                                instance_dict.update({value:dict_seq.get(str(num_instance))})
                                kitti_mask[indx] = dict_seq.get(str(num_instance))
                else:
                    kitti_mask[x,y] = dict_seq.get(str(num_instance))

            old_num_frame = num_frame

    res_im = Image.fromarray(kitti_mask.astype(np.uint32))
    resize = transforms.Resize(final_shape, interpolation=Image.NEAREST)
    res_im = resize(res_im)

    res_im.save(submission_dir + '/%06d.png' % (num_frame))

    kitti_mask = np.zeros(pred_mask.shape)
