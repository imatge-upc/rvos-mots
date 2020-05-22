########################################################################
# Interface for accessing the KITTI-MOTS dataset.
########################################################################

from collections import namedtuple

import numpy as np

from PIL import Image
from .base_kittimots import Sequence, SequenceClip, Annotation, AnnotationClip, BaseLoader, Segmentation, SequenceClip_simple, \
    AnnotationClip_simple
from misc.config_kittimots import cfg, phase, db_read_sequences
from .transforms.transforms import RandomAffine

from easydict import EasyDict as edict
import lmdb
import os.path as osp
import time
import glob
from .dataset import MyDataset


class KITTIMOTSLoader(MyDataset):
    """
    Helper class for accessing the KITTI-MOTS dataset.

    Arguments:
      phase         (string): dataset set eg. train, val. (See config.phase)
      single_object (bool):   assign same id (==1) to each object.

    Members:
      sequences (list): list of 'Sequence' objects containing RGB frames.
      annotations(list): list of 'Annotation' objects containing ground-truth segmentations.
    """

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split='train',
                 e=0,
                 resize=False,
                 inputRes=None,
                 video_mode=True,
                 use_prev_mask=False,
                 eval=False):

        self._phase = split
        self._single_object = args.single_object
        self._length_clip = args.length_clip
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.e = e
        self.inputRes = inputRes
        self.video_mode = video_mode
        self.max_seq_len = args.gt_maxseqlen
        self.dataset = args.dataset
        self.flip = augment
        self.use_prev_mask = use_prev_mask
        self.eval = eval

        if augment:
            if self._length_clip == 1:
                self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                           translation_range=args.translation,
                                                           shear_range=args.shear,
                                                           zoom_range=(args.zoom, max(args.zoom * 2, 1.0)),
                                                           interp='nearest')
            else:
                self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                           translation_range=args.translation,
                                                           shear_range=args.shear,
                                                           zoom_range=(args.zoom, max(args.zoom * 2, 1.0)),
                                                           interp='nearest',
                                                           lazy=True)

        else:
            self.augmentation_transform = None

        self._db_sequences = db_read_sequences(self._phase)

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time. It is recommended to use LMDB.')

        # Load sequences
        self.sequences = [Sequence(self._phase, s.name, lmdb_env=lmdb_env_seq) for s in self._db_sequences]

        self._db_sequences = db_read_sequences(self._phase)

        # Load annotations
        self.annotations = [Annotation(self._phase, s.name, self._single_object, lmdb_env=lmdb_env_annot)
                            for s in self._db_sequences]

        # Load sequences
        self.sequence_clips = []

        self._db_sequences = db_read_sequences(self._phase)
        for seq, s in zip(self.sequences, self._db_sequences):

            if self.use_prev_mask == False:

                images = seq.files

                starting_frame_idx = 0
                starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
                self.sequence_clips.append(SequenceClip_simple(seq, starting_frame))
                num_frames = self.sequence_clips[-1]._numframes
                num_clips = int(num_frames / self._length_clip)

                for idx in range(num_clips - 1):
                    starting_frame_idx += self._length_clip
                    starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
                    self.sequence_clips.append(SequenceClip_simple(seq, starting_frame))

            else:

                annot_seq_dir = osp.join(cfg.PATH.ANNOTATIONS, s.name)
                annotations = glob.glob(osp.join(annot_seq_dir, '*.png'))
                annotations.sort()
                # We only consider the first frame annotated to start the inference mode with such a frame
                starting_frame = int(osp.splitext(osp.basename(annotations[0]))[0])
                self.sequence_clips.append(SequenceClip(self._phase, s.name, starting_frame, lmdb_env=lmdb_env_seq))

        # Load annotations
        self.annotation_clips = []
        self._db_sequences = db_read_sequences(self._phase)
        for annot, s in zip(self.annotations, self._db_sequences):

            images = annot.files

            starting_frame_idx = 0
            starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
            self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))
            num_frames = self.annotation_clips[-1]._numframes
            num_clips = int(num_frames / self._length_clip)

            for idx in range(num_clips - 1):
                starting_frame_idx += self._length_clip
                starting_frame = int(osp.splitext(osp.basename(images[starting_frame_idx]))[0])
                self.annotation_clips.append(AnnotationClip_simple(annot, starting_frame))

        self._keys = dict(zip([s for s in self.sequences],
                              range(len(self.sequences))))

        self._keys_clips = dict(zip([s.name + str(s.starting_frame) for s in self.sequence_clips],
                                    range(len(self.sequence_clips))))

        try:
            self.color_palette = np.array(Image.open(
                self.annotations[0].files[0]).getpalette()).reshape(-1, 3)
        except Exception as e:
            self.color_palette = np.array([[0, 255, 0]])

    def get_raw_sample(self, key):
        """ Get sequences and annotations pairs."""
        if isinstance(key, str):
            sid = self._keys[key]
        elif isinstance(key, int):
            sid = key
        else:
            raise InputError()


        return edict({
            'images': self.sequences[sid],
            'annotations': self.annotations[sid]
        })

    def get_raw_sample_clip(self, key):
        """ Get sequences and annotations pairs."""
        if isinstance(key, str):
            sid = self._keys_clips[key]
        elif isinstance(key, int):
            sid = key
        else:
            raise InputError()

        return edict({
            'images': self.sequence_clips[sid],
            'annotations': self.annotation_clips[sid]
        })

    def sequence_name_to_id(self, name):
        """ Map sequence name to index."""
        return self._keys[name]

    def sequence_name_to_id_clip(self, name):
        """ Map sequence name to index."""
        return self._keys_clips[name]

    def sequence_id_to_name(self, sid):
        """ Map index to sequence name."""
        return self._db_sequences[sid].name

    def sequence_id_to_name_clip(self, sid):
        """ Map index to sequence name."""
        return self.sequence_clips[sid]

    def iternames(self):
        """ Iterator over sequence names."""
        for s in self._db_sequences:
            yield s

    def iternames_clips(self):
        """ Iterator over sequence names."""
        for s in self.sequence_clips:
            yield s

    def iteritems(self):
        return self.__iter__()
