from PIL import Image
import argparse
import numpy as np
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument("--whitelineloc", type=int, default=None)
parser.add_argument("--pngs", nargs="*")
parser.add_argument("--filename")
args = parser.parse_args()

for index, image in enumerate(args.pngs):
    if (index == 0):
        images = scipy.misc.imread(image)
    else:
        temp_image = scipy.misc.imread(image)
        images = np.concatenate((images, temp_image), axis=1)

if args.whitelineloc:
   images[27:28] = 255

final_image = Image.fromarray(images)
final_image.save(args.filename)
