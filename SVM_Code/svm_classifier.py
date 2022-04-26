import numpy as np
from helpers import progressbar
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import svm

'''
READ FIRST: Relationship Between Functions

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
pixels_per_cell = (8,8)
cells_per_block = (2,2)
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

    We will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    (or sklearn.cluster.MiniBatchKMeans if KMeans takes to long for you) to do this. 
    Note that this can take a VERY LONG TIME to complete (upwards of ten minutes 
    for large numbers of features and large max_iter), so set the max_iter argument
    to something low (we used 100) and be patient. You may also find success setting
    the "tol" argument (see documentation for details)
    '''    
    num_imgs = len(image_paths)

    # Extract hog features
    hog_features = []
    for i in progressbar(range(num_imgs), "Loading ...", num_imgs):
        im = imread(image_paths[i])
        if len(im.shape) == 3:
            im = rgb2gray(im)
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

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')
    num_imgs = len(image_paths)
    histograms = np.zeros((num_imgs, len(vocab)))
    for i in progressbar(range(num_imgs), "Loading ...", num_imgs):
        im = imread(image_paths[i])
        if len(im.shape) == 3:
            im = rgb2gray(im)
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
