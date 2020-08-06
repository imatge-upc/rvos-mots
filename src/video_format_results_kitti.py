import PIL.Image as Image
from PIL import ImageDraw
import numpy as np
import os
from args import get_parser
from utils.utils import make_dir
import json
import matplotlib.pyplot as plt

class AnnotationsGenerator:

    def __init__(self, ext='.png'):
        self.ext = ext
        self.colors = [(0, 255, 0), #verd
                       (255, 0, 0), #vermell
                       (0, 0, 255), #blau
                       (255, 0, 255), # rosa
                       (0, 255, 255), #turquesa
                       (255, 128, 0), #taronja
                       (153, 153, 0), #green caca
                       (51, 153, 255), #blau cel
                       (153, 153, 255), #lila clar
                       (102, 0, 102), #morado
                       (178, 102, 255),
                       (204, 0, 204),
                       (0, 102, 0),
                       (102, 0, 0),
                       #(51, 0, 0),
                       (0, 64, 0),
                       (128, 64, 0),
                       (0, 192, 0),
                       (128, 192, 0),
                       (0, 64, 128),
                       (224, 224, 192),
                       (154, 154, 154),
                       (45, 45, 45),
                       (190, 40, 190),
                       (30, 30, 240),
                       (155, 155, 155),
                       (100, 70, 0),
                       (224, 0, 192),
                       (156, 156, 156),
                       (5, 105, 105),
                       (157, 157, 157),
                       (0, 128, 0),
                       (128, 128, 0),
                       (0, 0 ,128),
                       (158, 158, 158),
                       (128, 0, 128),
                       (159, 159, 159),
                       (238, 108, 33),
                       (160, 160, 160)

        ]

    def generate_annotations_file(self, root_dir, results_dir):
        print(results_dir)
        list_dirs = os.listdir(results_dir)
        for d in list_dirs:
            #folder_dir = os.path.join(frames_dir, d)
            #sequences = ["0014"]
            folder_dir = os.path.join(results_dir, d)
            #folder_dir = os.path.join(annotations_dir, sequences[0])
            name_dir = os.path.join(results_dir, d + "_video")
            print(results_dir)
            print(d)
            print(name_dir)
            make_dir(name_dir)
            dict_colors = {}
            k = 0
            for f in sorted(os.listdir(folder_dir)):


                if f.endswith(self.ext):

                    image_file = os.path.join(folder_dir, f)
                    img = Image.open(image_file)
                    img_array = np.array(img)
                    obj_ids = np.unique(img_array)
                    axis_x = img_array.shape[0]
                    axis_y = img_array.shape[1]
                    colImage = Image.new('RGBA', (axis_y, axis_x), (255, 0, 0, 0))
                    draw = ImageDraw.Draw(colImage)

                    for x in range(axis_x):
                        for y in range(axis_y):

                            for z in obj_ids:
                                if img_array[x, y] == z:
                                    if img_array[x, y] == 0:
                                        break
                                    if img_array[x, y] == 10000:
                                        break
                                    if img_array[x, y] == 2002:
                                        break
                                    if img_array[x, y] == 2001:
                                        break
                                    '''if img_array[x, y] == 1006:
                                        draw.point((y, x), fill=(0, 0, 255))'''
                                    if str(z) not in dict_colors:
                                        dict_colors.update({str(z): k})
                                        k += 1
                                    draw.point((y, x), fill = self.colors[dict_colors[str(z)]])

                    frame_img = os.path.join("/mnt/gpid07/imatge/mgonzalez/databases/KITTIMOTS/PNGImages", d, f)
                    #frame_img = "/mnt/gpid07/imatge/mgonzalez/databases/YouTUBE-VOS/00000_image.jpg"
                    background = (Image.open(frame_img)).convert("RGBA")

                    paste_mask = colImage.split()[3].point(lambda i: i * 60 / 100.)
                    background.paste(colImage, (0,0), paste_mask)

                    name_file = os.path.join(name_dir, f)
                    background.save(name_file, "PNG")

        print(json.dumps(dict_colors))




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    from misc.config_kittimots import cfg

    frame_generator_annotations = AnnotationsGenerator(ext='.png')
    results_dir = os.path.join("/mnt/gpid07/imatge/mgonzalez/rvos/models", args.results_dir)
    frame_generator_annotations.generate_annotations_file(cfg.PATH.DATA, results_dir)
