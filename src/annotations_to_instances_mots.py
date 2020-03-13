import PIL.Image as Image
import numpy as np
import os
from args import get_parser
from utils.utils import make_dir


class AnnotationsGenerator:

    def __init__(self, ext='.png'):
        self.ext = ext

    def generate_annotations_file(self, root_dir, frames_dir, annotations_dir):
        coded_ann_in_dirs = os.listdir(frames_dir)
        print(annotations_dir)
        for d in coded_ann_in_dirs:
            folder_dir = os.path.join(frames_dir, d)
            name_dir = os.path.join(annotations_dir, d)
            make_dir(name_dir)
            for f in os.listdir(folder_dir):
                if f.endswith(self.ext):
                    image_file = os.path.join(folder_dir, f)
                    img = np.array(Image.open(image_file))
                    obj_ids = np.unique(img)
                    axis_x = img.shape[0]
                    axis_y = img.shape[1]

                    for x in range(axis_x):
                        for y in range(axis_y):
                            if img[x, y] == 0:
                                img[x, y] = 0
                            elif img[x, y] == 10000:
                                img[x, y] = 0

                            else:
                                k = 0
                                for z in obj_ids:
                                    if img[x, y] == z:
                                        img[x, y] == k
                                    k += 1
                    new_img = Image.fromarray(img)
                    name_file = os.path.join(name_dir, f)
                    new_img.save(name_file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    from misc.config_kittimots import cfg

    frame_generator_annotations = AnnotationsGenerator(ext='.png')
    frame_generator_annotations.generate_annotations_file(cfg.PATH.DATA, cfg.PATH.CODED_ANNOTATIONS, cfg.PATH.ANNOTATIONS)
