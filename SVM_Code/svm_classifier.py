import numpy as np
from helpers import progressbar
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize, rescale
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import svm

'''
Functions in this file can be classified into 3 groups based on their roles:
Group 1: feature-extracting functions
        a) get_tiny_images: 
            read in the images from the input paths and size down the images; 
            the output tiny images are used as features
        b) get_bags_of_words: 
            read in the images from the input paths and 
            turn each of the images into a histogram of oriented gradients (hog); 
            the output histograms are used as features
Group 2: supplementary function for get_bags_of_words (the second function in Group 1)
        build_vocabulary:
            read in the images from the input paths and build a vocabulary using the images using K-Means;
            the output vocabulary are fed into get_bags_of_words
            (Only need to run this function in main.py once)
Group 3: classification function
        svm_classify:
            implement many-versus-one linear SVM classifier

In main.py, we will run different functions in Group 1 e.g.
    i) get_tiny_images + svm_classify   
    ii) get_bags_of_words + svm_classify
    to do scene classification.

Read main.py for more details.
'''
pixels_per_cell = (4,4)
cells_per_block = (4,4)
def get_tiny_images(image_paths):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, images are resized to a very small
    square resolution (e.g. 16x16). Images are resized to squares
    while ignoring their aspect ratio. 
    
    Normalizing these tiny images will increase performance modestly.
    '''

    tiny_features = []
    for im_path in image_paths:
        im = imread(im_path)
        if len(im.shape) == 3:
            im = rgb2gray(im)
        tiny_im = resize(im, (16,16), anti_aliasing=True)
        tiny_features.append( np.reshape(tiny_im, 256) )

    return np.asarray(tiny_features)

def build_vocabulary(image_paths, vocab_size):
    '''
    This function samples HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    HOG features are generated using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    '''    
    num_imgs = len(image_paths)

    # Extract hog features
    hog_features = []
    for i in progressbar(range(num_imgs), "Loading ...", num_imgs):
        im = imread(image_paths[i])
        if len(im.shape) == 3:
            im = rgb2gray(im)
        im = rescale(im, 3/18, anti_aliasing=True)
        feature_vector = hog(im, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
        feature_vectors = feature_vector.reshape(-1, np.product(cells_per_block)*9)
        hog_features.append(feature_vectors)
    hog_features = np.concatenate(hog_features, axis=0)
    
    # Cluster features
    kmeans = KMeans(n_clusters=vocab_size, max_iter=100).fit(hog_features)

    return kmeans.cluster_centers_

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')
    num_imgs = len(image_paths)
    histograms = np.zeros((num_imgs, len(vocab)))
    for i in progressbar(range(num_imgs), "Loading ...", num_imgs):
        im = imread(image_paths[i])
        if len(im.shape) == 3:
            im = rgb2gray(im)
        im = rescale(im, 3/18, anti_aliasing=True)
        # Extract hog features
        feature_vector = hog(im, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
        feature_vectors = feature_vector.reshape(-1, np.product(cells_per_block)*9)
        # Determine the nearest word for each feature
        distances = cdist(feature_vectors, vocab, 'cosine')
        nearest_words = np.argpartition(distances, 1, axis=1)[:,:1]
        # Add words to histogram
        for word in nearest_words:
            histograms[i, word] += 1

    return histograms

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    an SVM classifier on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    '''

    classifier =  svm.SVC()
    classifier.fit(train_image_feats, train_labels)

    return classifier.predict(test_image_feats)
