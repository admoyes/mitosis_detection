import numpy as np
import h5py
from os import walk, path
import argparse
import scipy.io

# module accepts a root directory of the dataset and loops through each folder, extracting image data and image labels
# assumes that each image is in its own folder (as in the CRCHistoPhentotypes dataset)
class dataset:

    def __init__(self, root_directory):
        self.root_directory = root_directory

    def build(self):
        dirs = [x for x in walk(self.root_directory)]
        print("[INFO] got " + str(len(dirs)) + " images")
        for root, dirs, files in dirs:
            if len(files) == 2:
                # is valid
                if files[1].split(".")[-1] == "mat":
                    label_path = label_path = path.join(root, files[1])
                    image_path = path.join(root, files[0])
                else:
                    label_path = label_path = path.join(root, files[0])
                    image_path = path.join(root, files[1])

                print("[INFO] reading " + label_path)
                mat = scipy.io.loadmat(label_path)
                mitosis_pixels = mat["detection"]

                # we now have all pixels in the image which are part of a mitosis



ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", type=str, default='', help="(required) root directory of dataset")
args = vars(ap.parse_args())

print("[WELCOME] dataset builder")

# check if root arg is given
if len(args["root"]) == 0:
    print("[ERROR] root directory must be given")
    print("[INFO] exiting")
else:
    print("[INFO] root directory " + args["root"])
    ds = dataset(args["root"])
    print(ds.build())
