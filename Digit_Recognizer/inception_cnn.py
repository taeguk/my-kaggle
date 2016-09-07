import tensorflow as tf
import numpy as np
import input_data
import io_data


VERSION = "v0.1"
#kaggle train data


initializer = tf.contrib.layers.xavier_initializer()


def input_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt = 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))


def hidden_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))


def output_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.add(tf.matmul(X, W), b)


def conv_layer(X, shape, strides = [1, 1, 1, 1]):
    global conv_layer_cnt
    conv_layer_cnt += 1
    W = tf.get_variable("Conv_W" + str(conv_layer_cnt), shape = shape, initializer=initializer)
    return tf.nn.conv2d(X, W, strides=strides, padding="SAME")


def relu_layer(X):
    return tf.nn.relu(X)


def max_pooling_layer(X, ksize, strides, padding="SAME"):
    return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding)


def avg_pooling_layer(X, ksize, strides, padding="SAME"):
    return tf.nn.avg_pool(X, ksize=ksize, strides=strides, padding=padding)


def inception(X, depths):
    input_depth = X.get_shape()[-1]
    with tf.variable_scope('branch_0'):
        branch_0 = conv_layer(X, shape=[1, 1, input_depth, depths[0]])
    with tf.variable_scope('branch_1'):
        branch_1 = conv_layer(X, shape=[1, 1, input_depth, depths[1]])
        branch_1 = conv_layer(branch_1, shape=[3, 3, depths[1], depths[2]])
    with tf.variable_scope('branch_2'):
        branch_2 = conv_layer(X, shape=[1, 1, input_depth, depths[3]])
        branch_2 = conv_layer(branch_2, shape=[5, 5, depths[3], depths[4]])
    with tf.variable_scope('branch_3'):
        branch_3 = avg_pooling_layer(X, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1])
        branch_3 = conv_layer(branch_3, shape=[1, 1, input_depth, depths[5]])
    return tf.concat(3, [branch_0, branch_1, branch_2, branch_3])

def make_model(X, dropout_rate):
    # X's shape is [?, 768]

    end_points = {}

    # Construct Conv / ReLU / Pooling layers
    net = tf.reshape(X, shape=[-1, 28, 28, 1])   # [?, 28, 28, 1]
    end_points['input'] = net

    global conv_layer_cnt
    conv_layer_cnt = 0

    """
    net = inception(net, [8,  6,8,  8,12,  4])  # [?, 28, 28, 32]
    end_points['inception_1'] = net

    net = inception(net, [16,  12,16,  16,24,  8])  # [?, 28, 28, 64]
    end_points['inception_2'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 14, 14, 64]
    end_points['pooling_2'] = net

    net = inception(net, [32,  24,32,  32,48,  16])  # [?, 14, 14, 128]
    end_points['inception_3'] = net

    net = inception(net, [64,  48,64,  64,96,  32])  # [?, 14, 14, 256]
    end_points['inception_4'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 7, 7, 256]
    end_points['pooling_4'] = net

    net = inception(net, [128,  96,128,  128,192,  64])  # [?, 7, 7, 512]
    end_points['inception_5'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 3, 3, 512]
    end_points['pooling_5'] = net
    """

    net = inception(net, [8, 6, 8, 8, 12, 4])  # [?, 28, 28, 32]
    end_points['inception_1'] = net

    net = inception(net, [16, 12, 16, 16, 24, 8])  # [?, 28, 28, 64]
    end_points['inception_2'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 14, 14, 64]
    end_points['pooling_2'] = net

    net = inception(net, [32, 24, 32, 32, 48, 16])  # [?, 7, 7, 128]
    end_points['inception_3'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 7, 7, 128]
    end_points['pooling_3'] = net

    net = inception(net, [64, 48, 64, 64, 96, 32])  # [?, 7, 7, 256]
    end_points['inception_4'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 3, 3, 256]
    end_points['pooling_4'] = net


    # Construct Fully Connected Layers
    fc_X = tf.reshape(net, shape=[-1, 3*3*256])

    fc_L1 = input_layer(fc_X, 3*3*256, 625, dropout_rate)
    fc_L2 = hidden_layer(fc_L1, 625, 625, dropout_rate)
    fc_L3 = hidden_layer(fc_L2, 625, 625, dropout_rate)
    fc_L4 = hidden_layer(fc_L3, 625, 625, dropout_rate)
    fc_L4S = tf.add(fc_L4, fc_L1)
    fc_L5 = output_layer(fc_L4S, 625, 10, dropout_rate)

    last_fc_L = fc_L5

    return last_fc_L

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

train_x_data, train_y_data = mnist.train.images, mnist.train.labels
train_data_len = len(train_x_data)

"""
train_x_data, train_y_data = io_data.get_train_data(one_hot = True)
train_data_len = len(train_x_data)
"""
test_x_data, test_y_data = mnist.test.images, mnist.test.labels
test_data_len = len(test_x_data)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

dropout_rate = tf.placeholder("float")

model = make_model(X, dropout_rate)
model_with_softmax = tf.nn.softmax(model)

# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

LEARNING_RATE = 0.001
learning_rate = tf.Variable(LEARNING_RATE)
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    """
        Variables and functions about
        Loading and Saving Data.
    """
    saver = tf.train.Saver()
    SAVE_DIR = 'save_files'
    import os
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    MODEL_SAVE_PATH = "{0}/{1}.{2}.ckpt".format(SAVE_DIR, os.path.basename(__file__), VERSION)
    INFO_FILE_PATH = "{0}/{1}.{2}.info".format(SAVE_DIR, os.path.basename(__file__), VERSION)

    def do_load():
        start_epoch = 1
        try:
            epochs = []
            avg_costs = []
            avg_accuracys = []
            learning_rates = []

            with open(INFO_FILE_PATH, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    data = line.split()
                    epochs.append(int(data[0]))
                    avg_costs.append(float(data[1]))
                    avg_accuracys.append(float(data[2]))
                    learning_rates.append(float(data[3]))
            saver.restore(sess, MODEL_SAVE_PATH)
            print("[*] The save file exists!")

            print("Do you wanna continue? (y/n) ", end="", flush=True)
            if input() == 'n':
                print("not continue...")
                print("[*] Start a training from the beginning.")
                os.remove(INFO_FILE_PATH)
                os.remove(MODEL_SAVE_PATH)
                sess.run(init)
            else:
                print("continue...")
                print("[*] Start a training from the save file.")
                start_epoch = epochs[-1] + 1
                for epoch, avg_cost, avg_accuracy, learning_rate in zip(epochs, avg_costs, avg_accuracys,
                                                                        learning_rates):
                    print("Epoch {0} with learning rate = {1} : avg_cost = {2}, avg_accuracy = {3}".
                          format(epoch, learning_rate, avg_cost, avg_accuracy))

        except FileNotFoundError:
            print("[*] There is no save files.")
            print("[*] Start a training from the beginning.")
            sess.run(init)

        return start_epoch

    def do_save():
        print("[progress] Saving result! \"Never\" exit!!", end='', flush=True)
        saver.save(sess, MODEL_SAVE_PATH)
        with open(INFO_FILE_PATH, "a") as f:
            f.write("{0} {1} {2} {3}".format(epoch, avg_cost, avg_accuracy, LEARNING_RATE) + os.linesep)
        print("", end='\r', flush=True)


    """
        Variables and functions about
        Training and Testing Model
    """
    DISPLAY_SAVE_STEP = 1
    TRAINING_EPOCHS = 1000
    BATCH_SIZE = 256

    def do_train():
        print("[progress] Training model for optimizing cost!", end='', flush=True)
        # Loop all batches for training
        avg_cost = 0
        for start in range(0, train_data_len, BATCH_SIZE):
            end = min(start + BATCH_SIZE, train_data_len)
            batch_x = train_x_data[start:end]
            batch_y = train_y_data[start:end]
            data = {X: batch_x, Y: batch_y, dropout_rate: 0.5}
            sess.run(train, feed_dict=data)
            avg_cost += sess.run(cost, feed_dict=data) * len(batch_x) / train_data_len

        print("", end='\r', flush=True)
        return avg_cost

    def do_test():
        print("[progress] Testing model for evaluating accuracy!", end='', flush=True)
        correct_prediction = tf.equal(tf.argmax(model_with_softmax, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Loop all batches for test
        avg_accuracy = 0
        for start in range(0, test_data_len, BATCH_SIZE):
            end = min(start + BATCH_SIZE, test_data_len)
            batch_x = test_x_data[start:end]
            batch_y = test_y_data[start:end]
            data = {X: batch_x, Y: batch_y, dropout_rate: 1.0}
            avg_accuracy += accuracy.eval(data) * len(batch_x) / test_data_len

        print("", end='\r', flush=True)
        return avg_accuracy


    ##### Start of flow

    start_epoch = do_load()

    if start_epoch == 1:
        avg_accuracy = do_test()
        print("After initializing, accuracy = {0}".format(avg_accuracy))

    # Training cycle
    for epoch in range(start_epoch, TRAINING_EPOCHS + 1):

        avg_cost = do_train()

        # Logging the result
        if epoch % DISPLAY_SAVE_STEP == start_epoch % DISPLAY_SAVE_STEP or epoch == TRAINING_EPOCHS:
            avg_accuracy = do_test()
            do_save()

            # Print Result
            print("Epoch {0} : avg_cost = {1}, accuracy = {2}".format(epoch, avg_cost, avg_accuracy))

    # Save Result to submission
    print("Save Result to submission...")
    real_test_x = io_data.get_test_data()
    real_test_data_len = len(real_test_x)
    real_pred_y = np.array([])

    for start in range(0, real_test_data_len, BATCH_SIZE):
        end = min(start + BATCH_SIZE, real_test_data_len)
        batch_x = real_test_x[start:end]

        data = {X: batch_x, dropout_rate: 1.0}
        pred = sess.run((tf.argmax(model_with_softmax, 1)), feed_dict = {X: batch_x, dropout_rate: 1.0})
        real_pred_y = np.concatenate([real_pred_y, pred])

    real_pred_y = real_pred_y.astype(int)
    io_data.save_result(real_pred_y)