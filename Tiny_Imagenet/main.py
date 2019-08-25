import tensorflow as tf
import numpy as np
from models import *
from argparse import ArgumentParser
import os

print('Tensorflow version: {}'.format(tf.__version__))

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
train_batch_size = 100
test_batch_size = 20
epochs = 200
steps_per_epoch = int(100000/train_batch_size)
model_info = "##########\nModel: " + args.model + "\nTraining batch size: " + str(train_batch_size) + \
            "\nTesting batch size: "+ str(test_batch_size) + "\nEpochs: " + str(epochs)

def _parse_function_train(example_proto, augmentation=False):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    target = parsed_features['target']

    if augmentation:
        image = tf.image.resize_image_with_crop_or_pad(image, 70, 70)
        image = tf.random_crop(image, size = [64, 64, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.8, 1.25)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.25)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, target

def dataset_trans(dataset, is_train_data):
    if is_train_data:
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(train_batch_size)
    else:
        dataset = dataset.repeat()
        dataset = dataset.batch(test_batch_size)
    return dataset


def to_tensor():
    #training data
    train_dataset = tf.data.TFRecordDataset('./train.tfrecords')
    train_dataset = train_dataset.map(lambda x: _parse_function_train(x, True))
    print(train_dataset.output_types)
    print(train_dataset.output_shapes)
    #testing data
    test_dataset = tf.data.TFRecordDataset('./val.tfrecords')
    test_dataset = test_dataset.map(lambda x: _parse_function_train(x))
    
    train_dataset = dataset_trans(train_dataset, True)
    test_dataset = dataset_trans(test_dataset, False)
    
    #make feedable iterator 
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    feature_batch, label_batch = iterator.get_next()
    
    #make one shot iterator
    training_iterator = train_dataset.make_one_shot_iterator()
    testing_iterator = test_dataset.make_one_shot_iterator()

    #onehot
    label_batch = tf.one_hot(label_batch, 200, axis=1, dtype=tf.float32)
    
    return handle, feature_batch, label_batch, training_iterator, testing_iterator

def prediction(inference, y_placeholder):
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y_placeholder, 1))
        correct_count = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_count)
    return accuracy

def lr_scheduling(global_step, learning_rate):
    if ((global_step == int(steps_per_epoch * epochs * 0.4)) |
       (global_step == int(steps_per_epoch * epochs * 0.6)) |
       (global_step == int(steps_per_epoch * epochs * 0.8))):
        return learning_rate*0.1

    elif global_step == int(steps_per_epoch * epochs * 0.9):
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

    g_1 = tf.Graph()
    with g_1.as_default():
        handle, X_placeholder, y_placeholder, training_iterator, testing_iterator = to_tensor()
        
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
                                        feed_dict = {handle: training_handle, learning_rate: lr, 
                                                     inv_learning_rate: lr * 2, is_training: True})
                lr = lr_scheduling(gs, lr)

                if gs % ((100000/train_batch_size) * 2) == 0.0:
                    train_accu.append(accu)
                    accu = 0
                    test_steps = int(10000/test_batch_size)
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