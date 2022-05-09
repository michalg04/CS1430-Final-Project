"""
Based on Brown University's
CS1430 - Computer Vision,
Scene Classification Homework Assignment (#4)
"""
#!/usr/bin/python
import numpy as np
import os
import argparse

from helpers import get_image_paths
from svm_classifier import get_tiny_images, build_vocabulary, get_bags_of_words, \
    svm_classify
from create_results_webpage import create_results_webpage


def projPenuomniaBoW(feature='placeholder', load_vocab='True',
                    data_path='../chest_xray/'):
    '''
    For this portion of the project, we evaluate performance for two
    different features representations and an SVM classifier.
        1) Tiny image features
        2) Bag of word features
    '''

    # Step 0: Set up parameters, category list, and image paths.
    FEATURE = feature
    CLASSIFIER = 'support_vector_machine'

    # This is the list of categories / directories to use. 
    categories = ['NORMAL', 'PNEUMONIA']

    # This list of shortened category names is used later for visualization.
    abbr_categories = ['N', 'P']

    # Number of training examples per category to use. This is also the number of
    # test cases per category as well.
    num_train_per_cat = 200

    # This function returns string arrays containing the file path for each train
    # and test image, as well as string arrays with the label of each train and
    # test image.
    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)

    ############################################################################
    ## Step 1: Represent each image with the appropriate feature
    # Each function to construct features returns an N x d matrix, where
    # N is the number of paths passed to the function and d is the
    # dimensionality of each image representation.
    ############################################################################

    print('Using %s representation for images.' % FEATURE)

    if FEATURE.lower() == 'tiny_image':
        print('Loading tiny images...')
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats  = get_tiny_images(test_image_paths)
        print('Tiny images loaded.')

    elif FEATURE.lower() == 'bag_of_words':
        # Because building the vocabulary takes a long time, we save the generated
        # vocab to a file and re-load it each time to make testing faster. To
        # re-generate the vocab, set --load_vocab to False.
        # This will re-compute the vocabulary.
        if load_vocab == 'True':
            # check if vocab exists
            if not os.path.isfile('vocab.npy'):
                print('IOError: No existing visual word vocabulary found. Please set --load_vocab to False.')
                exit()

        elif load_vocab == 'False':
            print('Computing vocab from training images.')

            #Larger values will work better (to a point), but are slower to compute
            vocab_size = 200

            vocab = build_vocabulary(train_image_paths, vocab_size)
            np.save('vocab.npy', vocab)
        else:
            raise ValueError('Unknown load flag! Should be boolean.')

        train_image_feats = get_bags_of_words(train_image_paths)

        test_image_feats  = get_bags_of_words(test_image_paths)

    elif FEATURE.lower() == 'placeholder':
        train_image_feats = []
        test_image_feats = []

    else:
        raise ValueError('Unknown feature type!')

    ############################################################################
    ## Step 2: Classify each test image by training and using the SVM.
    # Each function to classify test features will return an N x 1 string array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' will be one of the strings in 'categories',
    # 'train_labels', and 'test_labels'.
    ############################################################################

    print('Using %s classifier to predict test set categories.' % CLASSIFIER)
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    ############################################################################
    ## Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section.

    # This function will recreate results_webpage/index.html and various image
    # thumbnails each time it is called.
    ############################################################################

    create_results_webpage( train_image_paths, \
                            test_image_paths, \
                            train_labels, \
                            test_labels, \
                            categories, \
                            abbr_categories, \
                            predicted_categories)

if __name__ == '__main__':
    '''
    Command line usage:
    python main.py [-f | --feature <representation to use>]
                   [-v | --load_vocab <boolean>]
                   [-d | --data <data_filepath>]

    -f | --feature - flag - if specified, will perform scene recognition using
    either placeholder (placeholder), tiny image (tiny_image), or bag of words
    (bag_of_words) to represent scenes

    -v | --load_vocab - flag - Boolean; if (True), loads the existing vocabulary 
    stored in vocab.npy (under <ROOT>/code), else if (False), creates a new one.
    default=(True).
    
    -d | --data - flag - if specified, will use the provided file path as the
    location to the data directory. Defaults to ../chest_xray
    '''
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--feature', default='placeholder', help='Either placeholder, tiny_image, or bag_of_words')
    parser.add_argument('-v', '--load_vocab', default='True', help='Boolean for either loading existing vocab (True) or creating new one (False)')
    parser.add_argument('-d', '--data', default='../chest_xray', help='Filepath to the data directory')

    args = parser.parse_args()
    projPenuomniaBoW(args.feature, args.load_vocab, args.data)
