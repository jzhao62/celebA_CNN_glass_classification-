
from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from tools import *

###################################
### Import picture files
###################################

files_path = './data/train/'
files_path_test = './data/test/'

glassless_files_path = os.path.join(files_path, 'N*.jpg')
glass_files_path = os.path.join(files_path, 'Y*.jpg')

glassless_files_path_test = os.path.join(files_path_test, 'N*.jpg')
glass_files_path_test = os.path.join(files_path_test, 'Y*.jpg')

glassless_files = sorted(glob(glassless_files_path))
glass_files = sorted(glob(glass_files_path))

glassless_files_test = sorted(glob(glassless_files_path_test))
glass_files_test = sorted(glob(glass_files_path_test))

n_files = len(glassless_files) + len(glass_files)

n_files_test = len(glassless_files_test) + len(glass_files_test)

print("number of glassless: ", len(glassless_files))
print("number of glass: ", len(glass_files))

print("number of glassless for test", len(glassless_files_test))
print("number of glass for test", len(glass_files_test))
size_image = 64

print("added successfully")

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)

allX_test = np.zeros((n_files_test, size_image, size_image, 3), dtype='float64')
ally_test = np.zeros(n_files_test)

count = 0
count_test = 0

for f in glassless_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in glass_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

for f in glassless_files_test:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX_test[count_test] = np.array(new_img)
        ally_test[count_test] = 0
        count_test += 1
    except:
        continue

for f in glass_files_test:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX_test[count_test] = np.array(new_img)
        ally_test[count_test] = 1
        count_test += 1
    except:
        continue



###################################
# Prepare train & test samples
###################################

# test-train split
X_train, X_test, Y_train, Y_test = train_test_split(allX, ally, test_size=0.3, random_state=42)

# encode the Ys(label)
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)


###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 64x64 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained (use adam not sgd)
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)


# Wrap the network in a model object
model = tflearn.DNN(network)
print("Network build complete")

###################################
# Train model for 10 epochs
###################################
model.fit(X_train, Y_train, validation_set=(X_test, Y_test), batch_size=10, n_epoch=5, show_metric=True)


###################################
# Generate confusion matrix
###################################
def generate_confusion_matrix(y_score, y_true):
    a1 = np.array(y_score)
    a2 = np.array([0, 1])
    a3 = a1 * a2
    a4 = np.array(a3[:, 1])
    for i in range(0, len(a4)):
        if(a4[i]>0.5):
            a4[i] = 1
        if (a4[i] <= 0.5):
            a4[i] = 0

    class_names = ["glass", "glassless"]

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, a4)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()

y_score = model.predict(X_test)
y_true = Y_test.argmax(axis=1)
generate_confusion_matrix(y_score, y_true)


y_score_test = model.predict(allX_test)
generate_confusion_matrix(y_score_test, ally_test)



