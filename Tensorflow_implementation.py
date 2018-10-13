import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# Define path to TensorBoard log files
logPath = "./tb_logs_unbal4/"


#   Adds summaries statistics for use in TensorBoard visualization.
#      From https://www.tensorflow.org/get_started/summaries_and_tensorboard
def variable_summaries(var):
   with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

loaded_images = np.load(os.getcwd() + '/Data/slice80_norm.npy')
loaded_labels = np.load(os.getcwd() + '/Data/newlabels.npy')

train_images_norm = loaded_images[:-100]
train_labels = loaded_labels[:-100]
test_images_norm = loaded_images[-100:]
test_labels = loaded_labels[-100:]
os.getcwd() + '/Data/
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


# Using Interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data
with tf.name_scope("MRIdata_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 65536], name="x")
    y_ = tf.placeholder(tf.float32, [None, 2], name="y_")

# change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
#    which the Convolution NN can use.
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1,256,256,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

# We are using RELU as our activation function.  These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Convelution and Pooling - we do Convelution, and then pooling to control overfitting
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

# Define the Model

# 1st Convolution layer
with tf.name_scope('Conv1'):
    # 32 features for each 5X5 patch of the image
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], name="weight")
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32], name="bias")
        variable_summaries(b_conv1)
    # Do convolution on images, add bias and push through RELU activation
    conv1_wx_b = conv2d(x_image, W_conv1,name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)
    # take results and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

# 2nd Convolution layer
with tf.name_scope('Conv2'):
# Process the 32 features from Convolution layer 1, in 5 X 5 patch.  return 64 features weights and biases
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], name="weight")
        variable_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64], name="bias")
        variable_summaries(b_conv2)
    # Do convolution of the output of the 1st convolution layer.  Pool results
    conv2_wx_b = conv2d(h_pool1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name="pool")

with tf.name_scope('FC'):
    # Fully Connected Layer
    W_fc1 = weight_variable([64 * 64 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")
    #   Connect output of pooling layer 2 as input to full connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # get dropout probability as a training input.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Readout"):
# Readout layer
    W_fc2 = weight_variable([1024, 2], name="weight")
    b_fc2 = bias_variable([2], name="bias")

# Define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimization
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # What is correct
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# TB - Merge summaries
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# TB - Write the default graph out so we can view it's structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
import time

#  define number of steps and how often we display progress
num_steps = 50
display_every = 1

# Start timer
start_time = time.time()
end_time = time.time()
for i in range(num_steps):
    batch = next_batch(100,train_images_norm,train_labels)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
        # write summary to log
        tbWriter.add_summary(summary,i)

        print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))
        #     Accuracy on test data
        print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
            x: test_images_norm, y_: test_labels, keep_prob: 1.0})*100.0))


# Display summary
#     Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))
#     Accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: test_images_norm, y_: test_labels, keep_prob: 1.0})*100.0))
