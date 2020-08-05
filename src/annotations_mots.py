import PIL.Image as Image
import numpy as np
import os
from args import get_parser
from utils.utils import make_dir
import json


class AnnotationsGenerator:

    def __init__(self, ext='.png'):
        self.ext = ext
        self.colors = [(0, 255, 0),
                       (255, 0, 0),
                       (0, 0, 255),
                       (255, 0, 255),
                       (0, 255, 255),
                       (255, 128, 0),
                       (102, 0, 102),
                       (51, 153, 255),
                       (153, 153, 255),
                       (153, 153, 0),
                       (178, 102, 255),
                       (204, 0, 204),
                       (0, 102, 0),
                       (102, 0, 0),
                       (51, 0, 0),
                       (0, 64, 0),
                       (128, 64, 0),
                       (0, 192, 0),
                       (128, 192, 0),
                       (0, 64, 128),
                       (224, 224, 192),
                       (154, 154, 154),
                       (155, 155, 155),
                       (156, 156, 156),
                       (157, 157, 157),
                       (158, 158, 158),
                       (159, 159, 159),
                       (160, 160, 160),
                       (161, 161, 161)

        ]

    def generate_annotations_file(self, root_dir, frames_dir, annotations_dir):
        coded_ann_in_dirs = os.listdir(frames_dir)
        print(annotations_dir)
        #for d in coded_ann_in_dirs:
        #folder_dir = os.path.join(frames_dir, d)
        sequences = ["0018"]
        folder_dir = os.path.join(annotations_dir,"Coded_annots", sequences[0])
        name_dir = os.path.join(annotations_dir, sequences[0])
        make_dir(name_dir)
        dict_colors = {}
        k = 0
        for f in sorted(os.listdir(folder_dir)):

            if f.endswith(self.ext):
                image_file = os.path.join(folder_dir, f)
                img = np.array(Image.open(image_file))
                obj_ids = np.unique(img)
                axis_x = img.shape[0]
                axis_y = img.shape[1]
                colImage = np.zeros((axis_x, axis_y, 3), dtype="uint8")

                for x in range(axis_x):
                    for y in range(axis_y):
                        if img[x, y] == 0:
                            colImage[x, y, 0] = 0  # for red
                            colImage[x, y, 1] = 0  # for green
                            colImage[x, y, 2] = 0

                        elif img[x, y] == 10000:
                            colImage[x, y, 0] = 255  # for red
                            colImage[x, y, 1] = 255  # for green
                            colImage[x, y, 2] = 255

                        else:
                            for z in obj_ids:
                                if img[x, y] == z:
                                    if str(z) not in dict_colors:
                                        dict_colors.update({str(z): k})
                                        k += 1
                                    colImage[x, y, 0] = self.colors[dict_colors[str(z)]][0]  # for red
                                    colImage[x, y, 1] = self.colors[dict_colors[str(z)]][1]  # for green
                                    colImage[x, y, 2] = self.colors[dict_colors[str(z)]][2]

                colImage = Image.fromarray(colImage, mode='RGB')
                name_file = os.path.join(name_dir, f)
                colImage.save(name_file)
        print(json.dumps(dict_colors))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    from misc.config_kittimots import cfg

    frame_generator_annotations = AnnotationsGenerator(ext='.png')
    coded_annotations = "/mnt/gpid07/imatge/mgonzalez/rvos/models/0_evaluacio_qualitativa/Coded_annots/"
    annotations_dir = "/mnt/gpid07/imatge/mgonzalez/rvos/models/0_evaluacio_qualitativa/"
    #frame_generator_annotations.generate_annotations_file(cfg.PATH.DATA, cfg.PATH.CODED_ANNOTATIONS, cfg.PATH.ANNOTATIONS)
    frame_generator_annotations.generate_annotations_file(cfg.PATH.DATA, coded_annotations, annotations_dir)
