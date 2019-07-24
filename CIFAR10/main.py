import tensorflow as tf
import numpy as np
from models import *
from argparse import ArgumentParser
import os

print('Tensorflow version: {}'.format(tf.version.VERSION))

#parse args
helper = "Available models: MLP, CNN, ResNet_20, ResNet_32 and VGG. With AL model, add postfix '_AL' after model name."
parser = ArgumentParser()
parser.add_argument("-m", "--model", help = helper, 
                     choices=['MLP', 'MLP_AL', 'CNN', 'CNN_AL', 'ResNet_20', 'ResNet_20_AL', 'ResNet_32',
                                'ResNet_32_AL', 'VGG', 'VGG_AL'], 
                     required=True, dest = "model")
models = {'MLP': MLP, 'MLP_AL': MLP_AL, 'CNN': CNN, 'CNN_AL': CNN_AL, 'ResNet_20': ResNet_20, 
            'ResNet_20_AL': ResNet_20_AL, 'ResNet_32': ResNet_32, 'ResNet_32_AL': ResNet_32_AL,
             'VGG': VGG, 'VGG_AL': VGG_AL}
args = parser.parse_args()
model = models[args.model]

#global variables
train_batch_size = 32
test_batch_size = 20
epochs = 200
model_info = "##########\nModel: " + args.model + "\nTraining batch size: " + str(train_batch_size) + \
            "\nTesting batch size: "+ str(test_batch_size) + "\nEpochs: " + str(epochs)

def data_fetching():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') /255
    X_test = X_test.astype('float32') /255
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (X_train, y_train), (X_test, y_test)

def _parse_function_train(image):
    distorted_images = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    distorted_images = tf.random_crop(distorted_images, size = [32, 32, 3])
    distorted_images = tf.image.random_flip_left_right(distorted_images)
    distorted_images = tf.image.random_brightness(distorted_images, max_delta=63)
    distorted_images = tf.image.random_contrast(distorted_images, lower=0.2, upper=1.8)
    distorted_images = tf.image.per_image_standardization(distorted_images)
    return distorted_images

def _parse_function_test(image):
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

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
    train_dataset_X = train_dataset_X.map(_parse_function_train)

    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_y))
    
    #testing data
    test_dataset_X = tf.data.Dataset.from_tensor_slices(X_test)
    test_dataset_X = test_dataset_X.map(_parse_function_test)

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
    if ((global_step == 125000) |
       (global_step == 187500) |
       (global_step == 250000)):
        return learning_rate*0.1

    elif global_step == 281250:
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
        with tf.name_scope('is_training'):
            is_training = tf.placeholder(tf.bool)

        inference, train_steps, global_step, neuron_num = model(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training)

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
        #lr = 1e-3 only for "ResNet_XX" models
        if (args.model[:6] == 'ResNet') & (args.model[-3:] != '_AL'):
            lr = 1e-3
        else:
            lr = 1e-4

        print(model_info + "\nNeuron number: " + str(neuron_num) + "\n##########")
        try:
            while (True):
                __, gs, accu = sess.run([train_steps, global_step, accuracy], 
                                        feed_dict = {handle: training_handle, learning_rate: lr, inv_learning_rate: lr * 2, is_training: True})

                lr = lr_scheduling(gs, lr)

                if gs % ((len(X_train)/train_batch_size) * 2) == 0.0:
                    train_accu.append(accu)
                    accu = 0
                    test_steps = int(len(X_test)/test_batch_size)
                    for i in range(test_steps):
                        accu += sess.run(accuracy, feed_dict = {handle: testing_handle, is_training: False})
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