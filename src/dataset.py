from PIL import Image
import numpy as np
import h5py
from os import walk, path
import argparse
import scipy.io
import cv2
from sklearn.feature_extraction import image as skimage

# module accepts a root directory of the dataset and loops through each folder, extracting image data and image labels
# assumes that each image is in its own folder (as in the CRCHistoPhentotypes dataset)
class dataset:

    def __init__(self, root_directory, window_size):
        self.root_directory = root_directory
        self.window_size = window_size

    def build(self, showImages=False):
        walk_result = [x for x in walk(self.root_directory)]
        print("[INFO] got " + str(len(walk_result)) + " images")

        image_patches = []
        label_patches = []

        for root, dirs, files in walk_result:
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
                mitosis_pixels = np.array(mat["detection"]).astype(np.int) - 1
                # we now have all pixels in the image which are part of a mitosis
                # next we need to create a window for each pixel in the image and assign a class to it
                img = Image.open(image_path)
                img_array = np.array(img) # actual image as (500,500,3)
                label_array = np.zeros(img_array.shape)
                label_array[mitosis_pixels[:, 1], mitosis_pixels[:, 0], :] = (1, 1, 1)

                if showImages == True:
                    cv2.imshow("mitosis pixels", np.hstack((img_array/255, label_array)))
                    cv2.waitKey(0)

                current_img_patches = skimage.extract_patches_2d(img_array, (self.window_size, self.window_size))
                current_label_patches = skimage.extract_patches_2d(label_array, (self.window_size, self.window_size))
                print(current_img_patches.shape)
                #image_patches.append(current_img_patches)
                #label_patches.append(current_label_patches)

        np_image_patches = np.array(image_patches)
        np_label_patches = np.array(label_patches)
        print("[INFO] np_image_patches shape " + str(np_image_patches.shape))
        print("[INFO] np_label_patches shape " + str(np_label_patches.shape))

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", type=str, default='', help="(required) root directory of dataset")
ap.add_argument("-w", "--window", type=int, default=10, help="(optional) window size, defaults to 10")
args = vars(ap.parse_args())

print("[WELCOME] dataset builder")

# check if root arg is given
if len(args["root"]) == 0:
    print("[ERROR] root directory must be given")
    print("[INFO] exiting")
else:
    print("[INFO] root directory " + args["root"])
    ds = dataset(args["root"], args["window"])
    print(ds.build())
