import tensorflow as tf
import numpy as np
from models import *
from argparse import ArgumentParser
import os

print('Tensorflow version: {}'.format(tf.version.VERSION))

#parse args
helper = "Available models: MLP, CNN. With AL model, add postfix '_AL' after model name."
parser = ArgumentParser()
parser.add_argument("-m", "--model", help = helper, 
                     choices=['MLP', 'MLP_AL', 'CNN', 'CNN_AL'], 
                     required=True, dest = "model")
models = {'MLP': MLP, 'MLP_AL': MLP_AL, 'CNN': CNN, 'CNN_AL': CNN_AL}
args = parser.parse_args()
model = models[args.model]

#global variables
train_batch_size = 32
test_batch_size = 20
epochs = 200
model_info = "##########\nModel: " + args.model + "\nTraining batch size: " + str(train_batch_size) + \
            "\nTesting batch size: "+ str(test_batch_size) + "\nEpochs: " + str(epochs)

def data_fetching():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    X_train = X_train.astype('float32') /255
    X_test = X_test.astype('float32') /255
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (X_train, y_train), (X_test, y_test)

def dataset_trans(dataset, is_train_data):
    if is_train_data:
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(train_batch_size)
    else:
        dataset = dataset.repeat()
        dataset = dataset.batch(test_batch_size)
    return dataset


def to_tensor(X_train, X_test, y_train, y_test):
    #training data
    train_dataset_X = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_y))
    
    #testing data
    test_dataset_X = tf.data.Dataset.from_tensor_slices(X_test)
    test_dataset_y = tf.data.Dataset.from_tensor_slices(y_test)
    test_dataset = tf.data.Dataset.zip((test_dataset_X, test_dataset_y))
    
    train_dataset = dataset_trans(train_dataset, True)
    test_dataset = dataset_trans(test_dataset, False)
    
    #make feedable iterator 
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    feature_batch, label_batch = iterator.get_next()
    
    #make one shot iterator
    training_iterator = train_dataset.make_one_shot_iterator()
    testing_iterator = test_dataset.make_one_shot_iterator()
    
    return handle, feature_batch, label_batch, training_iterator, testing_iterator

def prediction(inference, y_placeholder):
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y_placeholder, 1))
        correct_count = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_count)
    return accuracy

def lr_scheduling(global_step, learning_rate):
    if ((global_step == 150000) |
       (global_step == 225000) |
       (global_step == 300000)):
        return learning_rate*0.1

    elif global_step == 337500:
        return learning_rate*0.5
    else:
        return learning_rate

def save_result(training_accu, testing_accu, session):
    if not os.path.exists('./accuracy'):
        os.makedirs('./accuracy')
    np.savetxt("accuracy/"+ args.model + "-training.csv", np.array(training_accu), delimiter=",")
    np.savetxt("accuracy/"+ args.model + "-testing.csv", np.array(testing_accu), delimiter=",")

    if not os.path.exists('./saver'):
        os.makedirs('./saver')
    saver = tf.train.Saver()
    saver.save(session, "./saver/"+ args.model + "/model.ckpt")

def main():
    (X_train, y_train), (X_test, y_test) = data_fetching()
    print('Training shape :', y_train.shape)
    print('Testing shape :', y_test.shape)

    g_1 = tf.Graph()
    with g_1.as_default():
        handle, X_placeholder, y_placeholder, training_iterator, testing_iterator = to_tensor(X_train, X_test, y_train, y_test)
        with tf.name_scope('learning_rate'):
            learning_rate = tf.placeholder(tf.float32)
        with tf.name_scope('inv_learning_rate'):
            inv_learning_rate = tf.placeholder(tf.float32)

        inference, train_steps, global_step, neuron_num = model(X_placeholder, y_placeholder, learning_rate, inv_learning_rate)

        accuracy = prediction(inference, y_placeholder)

        init = tf.global_variables_initializer()
        
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    config.gpu_options.allow_growth = True

    train_accu = []
    test_accu = []
    with tf.Session(graph = g_1, config = config) as sess:
        sess.run(init)
        #writer = tf.summary.FileWriter('logs/', sess.graph)
        
        training_handle = sess.run(training_iterator.string_handle())
        testing_handle = sess.run(testing_iterator.string_handle())
        
        step = 0
        gs = 0
        lr = 1e-4

        print(model_info + "\nNeuron number: " + str(neuron_num) + "\n##########")
        try:
            while (True):
                __, gs, accu = sess.run([train_steps, global_step, accuracy], 
                                        feed_dict = {handle: training_handle, learning_rate: lr, inv_learning_rate: lr * 2})

                lr = lr_scheduling(gs, lr)

                if gs % ((len(X_train)/train_batch_size) * 2) == 0.0:
                    train_accu.append(accu)
                    accu = 0
                    test_steps = int(len(X_test)/test_batch_size)
                    for i in range(test_steps):
                        accu += sess.run(accuracy, feed_dict = {handle: testing_handle})
                    test_accu.append(accu/test_steps)
                    print("Epoch :", step)
                    print("Learning rate :", lr)
                    print("Training accu :", train_accu[-1], "Testing accu :", test_accu[-1])
                    step += 2
        except:
            print('End of sequence :)')

        save_result(train_accu, test_accu, sess)

if __name__ == '__main__':
    main()