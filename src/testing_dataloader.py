import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from utils.utils import batch_to_var, batch_to_var_test, make_dir, outs_perms_to_cpu, load_checkpoint, check_parallel
from modules.model import RSISMask, FeatureExtractor
from test import test, test_prev_mask
from dataloader.dataset_utils import sequence_palette
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import toimage
#import scipy
from dataloader.dataset_utils import get_dataset
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import sys, os
import json
from torch.autograd import Variable
import time
import os.path as osp
import cv2

parser = get_parser()
args = parser.parse_args()

split = args.eval_split
dataset = args.dataset
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

image_transforms = transforms.Compose([to_tensor, normalize])

dataset = get_dataset(args,
                      split=split,
                      image_transforms=image_transforms,
                      target_transforms=None,
                      augment=args.augment and split == 'train',
                      inputRes=(256, 448),
                      video_mode=True,
                      use_prev_mask=True)

loader = data.DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              drop_last=False)


for batch_idx, (inputs, targets,seq_name,starting_frame) in enumerate(loader):

    for ii in range(len(targets)):
        x, y_mask, sw_mask = batch_to_var(args, inputs[ii], targets[ii])
        mask_pred = (torch.squeeze(y_mask)).cpu().numpy()
        mask_pred = mask_pred.astype(np.uint8)
        mask_pred = Image.fromarray(mask_pred)
        base_dir = '/mnt/gpid07/imatge/mgonzalez/databases/Prova/'
        dir = osp.join(base_dir + str(batch_idx) + '_instance_' + str(ii) +'.png')
        #print(dir)
        mask_pred.save(dir)
