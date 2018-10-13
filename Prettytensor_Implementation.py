import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

num_iterations = 20
logPath = "./tb_logs_{}/".format(num_iterations)

# load
loaded_images = np.load(os.getcwd() + '/Data/balanced_fn/balanced_fn_80.npy')
loaded_labels = np.load(os.getcwd() + '/Data/balanced_fn/balanced_fn_labels.npy')

# split
train_images = loaded_images[:-100]
train_labels = loaded_labels[:-100]
data_validation_images = loaded_images[-100:-50]
data_validation_labels = loaded_labels[-100:-50]
data_test_images = loaded_images[-50:]
data_test_labels = loaded_labels[-50:]

# helper functions
data_test_cls = []
for i in range(len(data_test_labels)):
    if np.array_equal(data_test_labels[i],data_test_labels[0]):
        data_test_cls.append(0) # 0 means normal
    else:
        data_test_cls.append(1)

data_validation_cls = []
for i in range(len(data_validation_labels)):
    if np.array_equal(data_validation_labels[i],data_validation_labels[0]):
        data_validation_cls.append(0) # 0 means normal
    else:
        data_validation_cls.append(1)

# The MRI Images are 256 by 256 in length
img_size = 256

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes (Yes or No)
num_classes = 2

# Plotting function to visualise the images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Call to see the images

# # Get the first images from the test-set.
# images = data_test_images[0:9]
#
# # Get the true classes for those images.
# cls_true = data_test_cls[0:9]
#
# # Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)

# Tensorflow variable declaration
with tf.name_scope("MRIdata_Input"):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    tf.summary.image('input_img', x_image, 5)

# Tensorflow graph architecture

x_pretty = pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Helper function to visualise thw weight variables

# def get_weights_variable(layer_name):
#     # Retrieve an existing variable named 'weights' in the scope
#     # with the given layer_name.
#     # This is awkward because the TensorFlow function was
#     # really intended for another purpose.
#
#     with tf.variable_scope(layer_name, reuse=True):
#         variable = tf.get_variable('weights')
#
#     return variable
#
# weights_conv1 = get_weights_variable(layer_name='layer_conv1')
# weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Training
with tf.name_scope("loss_optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
with tf.name_scope("accuracy"):
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Helper function to save the data
saver = tf.train.Saver()
save_dir = 'checkpoint_dir/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

tf.summary.scalar("training_accuracy", accuracy)
summarize_all = tf.summary.merge_all()

# Tensorflow run
session = tf.Session()
tbWriter = tf.summary.FileWriter(logPath, session.graph)

def init_variables():
    session.run(tf.global_variables_initializer())

init_variables()

train_batch_size = 100
# Best validation accuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
require_improvement = 50

# Counter for total number of iterations performed so far.
total_iterations = 0

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    #np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = next_batch(train_batch_size,train_images,train_labels)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # session.run(optimizer, feed_dict=feed_dict_train)
        _, summary = session.run([optimizer, summarize_all], feed_dict=feed_dict_train)

        if (total_iterations % 1 == 0) or (i == (num_iterations - 1)):

            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation

                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''

            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 20


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.


        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = data_test_images,
                       labels = data_test_labels,
                       cls_true = data_test_cls)

def predict_cls_validation():
    return predict_cls(images = data_validation_images,
                       labels = data_validation_labels,
                       cls_true = data_validation_cls)

def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)

def print_test_accuracy(show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Performance before any optimization

print_test_accuracy()
# plot_conv_weights(weights=weights_conv1)

optimize(num_iterations)
print_test_accuracy(show_confusion_matrix=True)
# plot_conv_weights(weights=weights_conv1)

# Close the TensorFlow session
session.close()

# # To restore previously recorded variables
# init_variables()  # Ensures that the previously saved variables are not used
# saver.restore(sess=session, save_path=save_path)
# print_test_accuracy(show_confusion_matrix=True)
# plot_conv_weights(weights=weights_conv1)
