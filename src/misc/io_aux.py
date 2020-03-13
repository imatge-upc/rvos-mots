from PIL import Image
import numpy as np

from collections import defaultdict
import pprint

from misc.config_kittimots import cfg
#from misc.config import cfg
from args import get_parser


def imread_indexed(filename):
  """ Load image given filename."""
  # Dataset configuration initialization
  parser = get_parser()
  args = parser.parse_args()

  if args.dataset == 'kittimots':
    im = Image.open(filename).convert('P')
  else:
    im = Image.open(filename)

  annotation = np.atleast_3d(im)[...,0]
  reshape = np.array(im.getpalette()).reshape(-1, 3)

  return annotation, reshape

def imwrite_indexed(filename,array,color_palette=cfg.palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')