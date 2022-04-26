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
    for i in progressbar(range(image_paths.shape[0]), "Loading ...", image_paths.shape[0]):
    # for img_path in image_paths:
        img_path = image_paths[i]
        image = imread(img_path)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image = resize(image, (hp.image_size, hp.image_size))
        images.append(image)

    return np.array(images)

'''

'''
def get_data(train_path, val_path, test_path):

    # Get images paths
    train_img_paths, y_train = get_image_paths(train_path)
    val_img_paths, y_val = get_image_paths(val_path)
    test_img_paths, y_test = get_image_paths(test_path)

    # Shuffle training images
    indices = np.arange(train_img_paths.shape[0])
    np.random.shuffle(indices)
    train_img_paths = train_img_paths[indices]
    y_train = y_train[indices]

    # Read images
    x_train = read_images(train_img_paths)
    x_val = read_images(val_img_paths)
    x_test = read_images(test_img_paths)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape} - y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test

# get_data("../data/chest_xray/train", "../data/chest_xray/val", "../data/chest_xray/test")