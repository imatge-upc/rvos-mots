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

    def generate_annotations_file(self, root_dir, frames_dir, annotations_dir):
        coded_ann_in_dirs = os.listdir(frames_dir)
        print(annotations_dir)
        #for d in coded_ann_in_dirs:
        #folder_dir = os.path.join(frames_dir, d)
        sequences = ["0014"]
        folder_dir = os.path.join(annotations_dir,"Coded_annots", sequences[0])
        #folder_dir = os.path.join(annotations_dir, sequences[0])
        name_dir = os.path.join(annotations_dir, sequences[0])
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
                #colImage = Image.new('RGB', (axis_y, axis_x))
                draw = ImageDraw.Draw(colImage)


                '''dict_colors.update({str(1014): 0})
                dict_colors.update({str(1010): 1})
                dict_colors.update({str(1008): 2})
                dict_colors.update({str(1011): 3})
                dict_colors.update({str(1009): 4})
                dict_colors.update({str(1012): 5})
                dict_colors.update({str(1015): 6})
                dict_colors.update({str(1013): 7})
                dict_colors.update({str(1017): 8})
                dict_colors.update({str(1022): 9})
                dict_colors.update({str(1016): 10})'''

                '''dict_colors.update({str(1020): 0})
                dict_colors.update({str(1019): 1})
                dict_colors.update({str(1013): 2})
                dict_colors.update({str(1008): 3})'''

                '''dict_colors.update({str(1006): 5})
                dict_colors.update({str(1005): 1})
                dict_colors.update({str(1000): 2})
                dict_colors.update({str(1021): 3})
                dict_colors.update({str(1007): 4})
                dict_colors.update({str(1019): 7})
                dict_colors.update({str(1018): 6})
                dict_colors.update({str(1004): 0})'''

                '''dict_colors.update({str(1006): 0})
                dict_colors.update({str(1003): 1})
                dict_colors.update({str(1001): 2})
                dict_colors.update({str(1002): 3})
                dict_colors.update({str(1004): 4})
                dict_colors.update({str(1007): 5})
                dict_colors.update({str(1008): 6})'''

                '''dict_colors.update({str(1000): 0})
                dict_colors.update({str(1015): 1})
                dict_colors.update({str(1016): 2})
                dict_colors.update({str(1006): 3})
                dict_colors.update({str(1004): 4})
                dict_colors.update({str(1005): 5})
                dict_colors.update({str(1008): 6})
                dict_colors.update({str(1009): 7})
                dict_colors.update({str(1010): 8})
                dict_colors.update({str(1013): 9})
                dict_colors.update({str(1007): 10})
                dict_colors.update({str(1011): 11})
                dict_colors.update({str(1012): 12})
                dict_colors.update({str(1014): 13})'''

                '''dict_colors.update({str(1013): 0})
                dict_colors.update({str(1008): 1})
                dict_colors.update({str(1014): 2})
                dict_colors.update({str(1025): 3})'''

                '''dict_colors.update({str(1010): 2})
                dict_colors.update({str(1009): 0})
                dict_colors.update({str(1000): 1})
                dict_colors.update({str(1005): 5})
                dict_colors.update({str(1004): 3})
                dict_colors.update({str(1003): 4})'''

                '''dict_colors.update({str(1000): 0})
                dict_colors.update({str(1005): 9})
                dict_colors.update({str(1006): 7})
                dict_colors.update({str(1007): 8})
                dict_colors.update({str(1019): 6})
                dict_colors.update({str(1010): 13})
                #dict_colors.update({str(1009): 3})'''

                '''dict_colors.update({str(1006): 3})
                dict_colors.update({str(1003): 1})
                dict_colors.update({str(1001): 2})
                dict_colors.update({str(1004): 4})
                dict_colors.update({str(1002): 5})
                dict_colors.update({str(1007): 9})
                dict_colors.update({str(1008): 7})
                dict_colors.update({str(1011): 6})
                dict_colors.update({str(1013): 13})
                #dict_colors.update({str(1016): 12})
                dict_colors.update({str(1012): 8})'''

                '''dict_colors.update({str(1000): 0})
                dict_colors.update({str(1001): 1})
                dict_colors.update({str(1002): 2})
                dict_colors.update({str(1003): 3})
                dict_colors.update({str(1004): 4})
                dict_colors.update({str(1021): 5})
                dict_colors.update({str(1005): 9})
                dict_colors.update({str(1006): 7})
                dict_colors.update({str(1007): 8})
                dict_colors.update({str(1019): 6})
                dict_colors.update({str(1010): 13})
                dict_colors.update({str(1009): 3})'''

                '''dict_colors.update({str(1061): 1})
                dict_colors.update({str(1003): 2})
                dict_colors.update({str(1004): 3})
                dict_colors.update({str(1006): 4})
                dict_colors.update({str(1001): 5})'''


                for x in range(axis_x):
                    for y in range(axis_y):
                        '''if img_array[x, y] == 0:
                            colImage.append((255, 255, 255, 0))

                        elif img_array[x, y] == 10000:
                            colImage.append((255, 255, 255, 0))
                        else:'''


                        for z in obj_ids:
                            if img_array[x, y] == z:
                                if img_array[x, y] == 0:
                                    break
                                    #draw.point((y, x), fill=(0,0,0))
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
                                '''if img_array[x, y] == 1000 or img_array[x, y] == 1010 or img_array[x, y] == 1000 or img_array[x, y] == 1007 or img_array[x, y] == 1019:
                                    draw.point((y, x), fill=self.colors[dict_colors[str(z)]])'''
                                '''if img_array[x, y] == 1003 or img_array[x, y] == 1001 or img_array[x, y] == 1006 or  img_array[x, y] == 1004 or img_array[x, y] == 1002\
                                        or img_array[x, y] == 1007 or img_array[x, y] == 1008 or img_array[x, y] == 1011\
                                        or img_array[x, y] == 1012  or img_array[x, y] == 1013:
                                    draw.point((y, x), fill=self.colors[dict_colors[str(z)]])'''
                                '''if img_array[x, y] == 1000 or img_array[x, y] == 1001 or img_array[x, y] == 1002 or img_array[x, y] == 1003\
                                        or img_array[x, y] == 1004 or img_array[x, y] == 1021 or img_array[x, y] == 1005\
                                        or img_array[x, y] == 1006 or img_array[x, y] == 1007 or img_array[x, y] == 1019\
                                        or img_array[x, y] == 1010:
                                    draw.point((y, x), fill=self.colors[dict_colors[str(z)]])'''
                                '''if img_array[x, y] == 1061 or img_array[x, y] == 1003 or img_array[x, y] == 1004 or img_array[x, y] == 1006\
                                        or img_array[x, y] == 1001:
                                    draw.point((y, x), fill=self.colors[dict_colors[str(z)]])'''

                                '''if img_array[x, y] == 1027:
                                    draw.point((y, x), fill=self.colors[5])
                                if img_array[x, y] == 1028:
                                    draw.point((y, x), fill=self.colors[0])
                                else:
                                    draw.point((y, x), fill=self.colors[dict_colors[str(z)]])'''
                                '''if img_array[x, y] == 1010:
                                    draw.point((y, x), fill=self.colors[4])
                                if img_array[x, y] == 1008:
                                    draw.point((y, x), fill=self.colors[3])
                                if img_array[x, y] == 1014:
                                    draw.point((y, x), fill=self.colors[5])
                                if img_array[x, y] == 1012:
                                    draw.point((y, x), fill=self.colors[6])
                                if img_array[x, y] == 1015:
                                    draw.point((y, x), fill=self.colors[7])
                                if img_array[x, y] == 1013:
                                    draw.point((y, x), fill=self.colors[8])
                                if img_array[x, y] == 1017:
                                    draw.point((y, x), fill=self.colors[9])
                                if img_array[x, y] == 1022:
                                    draw.point((y, x), fill=self.colors[10])
                                if img_array[x, y] == 1016:
                                    draw.point((y, x), fill=self.colors[11])
                                if img_array[x, y] == 1021:
                                    draw.point((y, x), fill=self.colors[12])
                                if img_array[x, y] == 1023:
                                    draw.point((y, x), fill=self.colors[13])
                                if img_array[x, y] == 1020:
                                    draw.point((y, x), fill=self.colors[14])
                                if img_array[x, y] == 1024:
                                    draw.point((y, x), fill=self.colors[15])
                                if img_array[x, y] == 1025:
                                    draw.point((y, x), fill=self.colors[16])
                                if img_array[x, y] == 1026:
                                    draw.point((y, x), fill=self.colors[17])
                                if img_array[x, y] == 1027:
                                    draw.point((y, x), fill=self.colors[18])
                                if img_array[x, y] == 1028:
                                    draw.point((y, x), fill=self.colors[1])'''
                                #colImage.append(self.colors[dict_colors[str(z)]])



                frame_img = os.path.join("/mnt/gpid07/imatge/mgonzalez/databases/KITTIMOTS/PNGImages", sequences[0], f)
                #frame_img = "/mnt/gpid07/imatge/mgonzalez/databases/YouTUBE-VOS/00000_image.jpg"
                background = (Image.open(frame_img)).convert("RGBA")

                paste_mask = colImage.split()[3].point(lambda i: i * 60 / 100.)
                background.paste(colImage, (0,0), paste_mask)

                name_file = os.path.join(name_dir, f)
                ##colImage.save(name_file, "PNG")
                background.save(name_file, "PNG")

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
