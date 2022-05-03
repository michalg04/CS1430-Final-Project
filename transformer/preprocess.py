import os
import glob
import sys
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

import hyperparameters as hp

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


'''

'''
def get_image_paths(data_path):

    categories = ["NORMAL", "PNEUMONIA"]
    image_paths = []
    labels = []

    for _, cat in enumerate(categories):
        paths = glob.glob(os.path.join(data_path, cat, '*.jpeg'))
        image_paths += paths
        img_class = 0 if cat == "NORMAL" else 1
        labels += [img_class] * len(paths)

    return np.array(image_paths), np.array(labels)

'''

'''
def read_images(image_paths):
    images = []
    # for i in progressbar(range(image_paths.shape[0]), "Loading ...", image_paths.shape[0]):
    for img_path in image_paths:
    #    img_path = image_paths[i]
        image = imread(img_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image = resize(image, (hp.image_size, hp.image_size))
        images.append(image)

    return np.array(images)

'''

'''
def get_data(filepath):
    print("Loading data...")

    # Get images paths
    img_paths, labels = get_image_paths(filepath)

    # Shuffle training images
    indices = np.arange(img_paths.shape[0])
    np.random.shuffle(indices)
    img_paths = img_paths[indices]
    labels = labels[indices]

    # Read images
    x = read_images(img_paths)

    print("Done loading data.")
    print(f"X shape: {x.shape} - Y shape: {labels.shape}")

    return x, labels

# get_data("../data/chest_xray/train", "../data/chest_xray/val", "../data/chest_xray/test")
