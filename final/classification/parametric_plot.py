'''
Train CNNs on the CIFAR10/CIFAR100
Plots a parametric plot between SB and LB
minimizers demonstrating the relative sharpness
of the two minima.

Requirements:
- Keras (with Theano)
- Matplotlib
- Numpy

GPU run command:
    KERAS_BACKEND=theano python plot_parametric_plot.py --network C[1-4]
'''

from __future__ import print_function
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from AlexnetCNN import AlexnetCNN
from keras.datasets import cifar10, mnist
import keras.backend as K
import os.path


parser = argparse.ArgumentParser(description=
                '''This code first trains the user-specific network (C[1-4])
                using small-batch ADAM and large-batch ADAM, and then plots
                the parametric plot connecting the two minimizers
                illustrating the sharpness difference.''')
parser.add_argument('weights', nargs=2, help='weights help')
parser.add_argument('-d', '--dataset', help='dataset', required=True)
args = parser.parse_args()
print(args)
weight_file1, weight_file2 = args.weights
dataset_name = args.dataset
data_module = {'mnist': mnist, 'cifar10': cifar10}[dataset_name]


(X_train, y_train), (X_test, y_test) = data_module.load_data()
if dataset_name == "mnist":
  img_color, img_rows, img_cols = 1, 28, 28
else:
  img_color, img_rows, img_cols = 3, 32, 32

nb_classes = 10

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_color)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_color)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#X_train, y_train = sorted_batches(X_train, y_train)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

nn = AlexnetCNN()
nn.setFormat(img_rows, img_cols, img_color)
nn.setNumClasses(nb_classes)
nn.build_model()
nn.load_weights(weight_file1)


m1_solution  = [ K.get_value(p) for p in nn.model.trainable_weights ]

nn.reset()
nn.load_weights(weight_file2)

m2_solution = [ K.get_value(p) for p in nn.model.trainable_weights ]

# parametric plot data collection
# we discretize the interval [-1,2] into 25 pieces
alpha_range = numpy.linspace(-1, 2, 25)
data_for_plotting = numpy.zeros((25, 4))

i = 0
for alpha in alpha_range:
    for p in range(len(m1_solution)):
        K.set_value(nn.model.trainable_weights[p], 
                    m2_solution[p]*alpha + m1_solution[p]*(1-alpha))
    train_xent, train_acc = nn.model.evaluate(X_train, y_train,
                                           batch_size=1024, verbose=0)
    test_xent, test_acc = nn.model.evaluate(X_test, y_test,
                                         batch_size=1024, verbose=0)
    data_for_plotting[i, :] = [train_xent, train_acc, test_xent, test_acc]
    i += 1

# finally, let's plot the data
# we plot the XENT loss on the left Y-axis
# and accuracy on the right Y-axis
# if you don't have Matplotlib, simply print
# data_for_plotting to file and use a different plotter

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alpha_range, data_for_plotting[:, 0], 'b-')
ax1.plot(alpha_range, data_for_plotting[:, 2], 'b--')

ax2.plot(alpha_range, data_for_plotting[:, 1]*100., 'r-')
ax2.plot(alpha_range, data_for_plotting[:, 3]*100., 'r--')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.grid(b=True, which='both')
fn = os.path.basename(weight_file1)+'_' +os.path.basename(weight_file2) +'.pdf'
plt.savefig('Figures/'+ fn)
print('Plot save as ' + fn + ' in the Figures/ folder')
