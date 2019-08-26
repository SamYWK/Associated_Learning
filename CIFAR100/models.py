import tensorflow as tf
import os

class_num = 100

def autoencoders(y_placeholder, neuron_num, num_of_ae, inv_learning_rate, y_function):
    ###########
    #y-forward#
    ###########
    y_list = []
    y = y_placeholder
    for i in range(num_of_ae):
        y = tf.layers.dense(tf.stop_gradient(y), neuron_num, y_function, name = 'y_%d'%i,
                    kernel_initializer = tf.keras.initializers.he_normal())
        y_list.append(y)
    
    ###########
    #y-inverse#
    ###########
    i_weights = []
    i_biases = []
    rev_train_steps = []
    for i in range(num_of_ae - 1):
        y_inv = tf.layers.dense(y_list[num_of_ae - 1 - i], neuron_num, y_function, name = 'y_inv_%d'%(num_of_ae - 1 - i),
                    kernel_initializer = tf.keras.initializers.he_normal())
        i_weights.append(tf.get_default_graph().get_tensor_by_name(os.path.split(y_inv.name)[0] + '/kernel:0'))
        i_biases.append(tf.get_default_graph().get_tensor_by_name(os.path.split(y_inv.name)[0] + '/bias:0'))
        
        rev_loss = tf.losses.mean_squared_error(predictions = y_inv, labels = tf.stop_gradient(y_list[num_of_ae - 2 - i]))
        rev_train_steps.append(tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss))
    
    #1st layer
    y_inv = tf.layers.dense(y_list[0], class_num, y_function, name = 'y_inv_0',
                    kernel_initializer = tf.keras.initializers.he_normal())
    i_weights.append(tf.get_default_graph().get_tensor_by_name(os.path.split(y_inv.name)[0] + '/kernel:0'))
    i_biases.append(tf.get_default_graph().get_tensor_by_name(os.path.split(y_inv.name)[0] + '/bias:0'))
    
    rev_loss = tf.losses.mean_squared_error(predictions = y_inv, labels = y_placeholder)
    global_step = tf.train.get_or_create_global_step()
    rev_train_steps.append(tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss, global_step = global_step))
    return rev_train_steps, global_step, y_list, i_weights, i_biases

def al_loss(inputs, y_list, index, name_scope, train_steps, update_opss, learning_rate, bridge_function, neuron_num, regularizer = None):
    flat = tf.layers.Flatten()(inputs)
    dense = tf.layers.dense(flat, 5*neuron_num, activation = bridge_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer)
    dense = tf.layers.dense(dense, neuron_num, activation = bridge_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer)
    
    #check is namescope a list or not
    if type(name_scope) == list:
        l2_loss = tf.losses.get_regularization_loss(name_scope[0])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope[0])
        update_opss.append(update_ops)
        for i in range(1, len(name_scope)):
            l2_loss += tf.losses.get_regularization_loss(name_scope[i])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope[i])
            update_opss.append(update_ops)
    else:
        l2_loss = tf.losses.get_regularization_loss(name_scope)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
        update_opss.append(update_ops)

    loss = tf.losses.mean_squared_error(predictions = dense, labels = tf.stop_gradient(y_list[index])) + l2_loss
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_steps.append(train_step)
    return dense

def conv_layer_bn(inputs, filters, regularizer, is_training, f_function, strides = 1, name_scope = None, is_batchnorm = True):
    output = tf.layers.conv2d(
                inputs = inputs,
                filters = filters,
                kernel_size = 3,
                padding = 'same',
                strides = strides,
                activation = None,
                kernel_regularizer = regularizer,
                kernel_initializer = tf.keras.initializers.he_normal(),
                name = name_scope)
    if is_batchnorm:
        output = tf.layers.batch_normalization(output, training = is_training)
    return f_function(output)

def resBlock(inputs, out_channels, strides, regularizer, is_training, f_function):
    conv1 = conv_layer_bn(inputs, out_channels, regularizer, is_training, f_function, strides)

    conv2 = conv_layer_bn(conv1, out_channels, regularizer, is_training, f_function, 1)
    
    
    if (strides != 1):
        inputs = tf.layers.conv2d(
                    inputs = inputs,
                    filters = out_channels,
                    kernel_size = 1,
                    padding = 'same',
                    strides = strides,
                    activation = None,
                    kernel_regularizer = regularizer, kernel_initializer = tf.initializers.he_normal())
    
    return f_function(tf.add(conv2, inputs))

def ResNet(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, learning_rate, inv_learning_rate, f_function, is_training):
    regularizer = tf.contrib.layers.l2_regularizer(scale = regulariztion_scale)
    stride_list = [1, 2, 2]
    num_filter = 16
    
    #forward
    with tf.name_scope('Layer_0'):
        net = tf.layers.conv2d(
                    inputs = X_placeholder,
                    filters = 16,
                    kernel_size = 3,
                    padding = 'same',
                    strides = 1,
                    activation = None,
                    kernel_regularizer = regularizer,
                    kernel_initializer = tf.initializers.he_normal())
        net = tf.layers.batch_normalization(net, training = is_training)
        net = f_function(net)
        
    resblock = net
    layer_count = 1
    for stride in stride_list:
        strides = [stride] + [1]*(num_blocks - 1)
        with tf.name_scope('Layer_' + str(layer_count)):
            for stride in strides:
                resblock = resBlock(resblock, num_filter, stride, regularizer, is_training, f_function)
        num_filter *= 2
        layer_count += 1
    
    pool = tf.layers.average_pooling2d(resblock, pool_size = 8, strides=8)
    flat = tf.layers.Flatten()(pool)
    #addition layer for bridge
    with tf.name_scope('Dense_0'):
        net = tf.layers.dense(flat, 5*neuron_num, kernel_initializer = tf.initializers.he_normal(),
                                        kernel_regularizer = regularizer)
        net = f_function(tf.layers.batch_normalization(net, training = is_training))
    with tf.name_scope('Dense_1'):
        net = tf.layers.dense(net, neuron_num, kernel_initializer = tf.initializers.he_normal(),
                                        kernel_regularizer = regularizer)
        net = f_function(tf.layers.batch_normalization(net, training = is_training))
    #end of addition
    for i in range(2, 5):
        with tf.name_scope('Dense_' + str(i)):
            net = tf.layers.dense(net, neuron_num, kernel_initializer = tf.initializers.he_normal(),
                                        kernel_regularizer = regularizer)
            net = f_function(tf.layers.batch_normalization(net, training = is_training))
    net = tf.layers.dense(net, class_num, kernel_initializer = tf.initializers.he_normal(),
                                    kernel_regularizer = regularizer)

    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = net)) + l2_loss
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    train_steps = tf.group([train_steps, update_ops])
    return net, train_steps, global_step, neuron_num

def ResNet_AL(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, learning_rate, inv_learning_rate, acti_functions, is_training):
    regularizer = tf.contrib.layers.l2_regularizer(scale = regulariztion_scale)
    stride_list = [1, 2, 2]
    num_filter = 16
    num_of_ae = 4
    train_steps = []
    update_opss = []
    
    f_function, y_function, bridge_function = acti_functions

    #autoencoders
    rev_train_steps, global_step, target_list, i_weights, i_biases = \
        autoencoders(y_placeholder, neuron_num, num_of_ae, inv_learning_rate, y_function)

    #forward
    with tf.name_scope('Layer_0'):
        net = tf.layers.conv2d(
                    inputs = X_placeholder,
                    filters = 16,
                    kernel_size = 3,
                    padding = 'same',
                    strides = 1,
                    activation = None,
                    kernel_regularizer = regularizer,
                    kernel_initializer = tf.initializers.he_normal())
        net = tf.layers.batch_normalization(net, training = is_training)
        net = f_function(net)

    with tf.name_scope('X_Losses_0'):
        al_loss(net, target_list, 0, 'Layer_0', train_steps, update_opss, learning_rate, bridge_function,
                neuron_num , regularizer)
        
    resblock = net
    layer_count = 1
    for stride in stride_list:
        strides = [stride] + [1]*(num_blocks - 1)

        name_scope = 'Layer_' + str(layer_count)

        resblock = tf.stop_gradient(resblock)
        with tf.name_scope(name_scope):
            for stride in strides:
                resblock = resBlock(resblock, num_filter, stride, regularizer, is_training, f_function)

        with tf.name_scope('X_Losses_' + str(layer_count)):
            dense = al_loss(resblock, target_list, layer_count, name_scope, train_steps, update_opss, 
                            learning_rate, bridge_function, neuron_num , regularizer)
        num_filter *= 2
        layer_count += 1
    
    all_group_train_steps = tf.group(train_steps + rev_train_steps + update_opss)
    #inference
    with tf.name_scope('inference'):
        for i in range(num_of_ae):
            dense = y_function(tf.matmul(dense, i_weights[i]) + i_biases[i])
    return dense, all_group_train_steps, global_step, neuron_num

def MLP(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    f_function = tf.nn.relu
    neuron_num = 3000

    net = tf.layers.Flatten()(X_placeholder)

    net = tf.layers.dense(net, neuron_num, f_function, name = 'Layer_0', 
                            kernel_initializer = tf.keras.initializers.he_normal())
    
    net = tf.layers.dense(net, neuron_num, f_function, name = 'Layer_1', 
                            kernel_initializer = tf.keras.initializers.he_normal())
    
    #addition layer for bridge
    net = tf.layers.dense(net, 5*neuron_num, f_function, name = 'Layer_2', 
                            kernel_initializer = tf.keras.initializers.he_normal())
    
    net = tf.layers.dense(net, neuron_num, f_function, name = 'Layer_3', 
                            kernel_initializer = tf.keras.initializers.he_normal())
    #end of addition

    net = tf.layers.dense(net, neuron_num, f_function, name = 'Layer_4', 
                            kernel_initializer = tf.keras.initializers.he_normal())
    
    net = tf.layers.dense(net, class_num, None, name = 'Layer_5', 
                            kernel_initializer = tf.keras.initializers.he_normal())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=net))
    global_step = tf.train.get_or_create_global_step()
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    return net, train_steps, global_step, neuron_num

def MLP_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    f_function = tf.nn.elu
    bridge_function = tf.nn.sigmoid
    y_function = tf.nn.sigmoid
    neuron_num = 3000

    num_of_ae = 2
    train_steps = []
    update_opss = []

    rev_train_steps, global_step, target_list, i_weights, i_biases = \
        autoencoders(y_placeholder, neuron_num, num_of_ae, inv_learning_rate, y_function)

    net = tf.layers.Flatten()(X_placeholder)

    for i in range(num_of_ae):
        net = tf.layers.dense(tf.stop_gradient(net), neuron_num, f_function, name = 'Layer_' + str(i+1), 
                                kernel_initializer = tf.keras.initializers.he_normal())
        
        dense = al_loss(net, target_list, i, 'Layer_' + str(i+1), train_steps, update_opss, learning_rate, bridge_function, neuron_num)

    all_group_train_steps = tf.group(train_steps + rev_train_steps + update_opss)

    #inference
    with tf.name_scope('inference'):
        for i in range(num_of_ae):
            dense = y_function(tf.matmul(dense, i_weights[i]) + i_biases[i])
    return dense, all_group_train_steps, global_step, neuron_num
    
def CNN(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    f_function = tf.nn.relu
    hidden_layer_count = 4
    neuron_num = 256
    regulariztion_scale = 0.0

    regularizer = tf.contrib.layers.l2_regularizer(scale = regulariztion_scale)

    #forward
    net = X_placeholder
    for i in range(hidden_layer_count-1):
        net = tf.layers.conv2d(
                inputs = net,
                filters = 32,
                kernel_size = 3,
                padding = 'same',
                activation = f_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer)

    net = tf.layers.conv2d(
            inputs = net,
            filters = 32,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = f_function,
            kernel_initializer = tf.keras.initializers.he_normal(),
            kernel_regularizer = regularizer)
    
    for i in range(hidden_layer_count-1):
        net = tf.layers.conv2d(
                inputs = net,
                filters = 64,
                kernel_size = 3,
                padding = 'same',
                activation = f_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer)
    
    net = tf.layers.conv2d(
            inputs = net,
            filters = 64,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = f_function,
            kernel_initializer = tf.keras.initializers.he_normal(),
            kernel_regularizer = regularizer)
    
    net = tf.layers.Flatten()(net)
    
    #addition layer for bridge
    net = tf.layers.dense(net, 5*neuron_num, activation = f_function, kernel_initializer = tf.keras.initializers.he_normal(),
                                kernel_regularizer = regularizer)
    
    net = tf.layers.dense(net, neuron_num, activation = f_function, kernel_initializer = tf.keras.initializers.he_normal(),
                                kernel_regularizer = regularizer)
    #end of addition
    for i in range(hidden_layer_count - 1):
        net = tf.layers.dense(net, neuron_num, activation = f_function, kernel_initializer = tf.keras.initializers.he_normal(),
                                kernel_regularizer = regularizer)
        
    net = tf.layers.dense(net, class_num, activation = None, kernel_initializer = tf.keras.initializers.he_normal(), 
                            kernel_regularizer = regularizer)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=net)) + tf.losses.get_regularization_loss()
    global_step = tf.train.get_or_create_global_step()
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    return net, train_steps, global_step, neuron_num

def CNN_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    f_function = tf.nn.elu
    bridge_function = tf.nn.sigmoid
    y_function = tf.nn.sigmoid
    hidden_layer_count = 4
    neuron_num = 256
    regulariztion_scale = 0.0

    regularizer = tf.contrib.layers.l2_regularizer(scale = regulariztion_scale)

    num_of_ae = 4
    train_steps = []
    update_opss = []

    rev_train_steps, global_step, target_list, i_weights, i_biases = \
        autoencoders(y_placeholder, neuron_num, num_of_ae, inv_learning_rate, y_function)

    #forward
    target_index = 0
    net = X_placeholder
    for i in range(hidden_layer_count - 1):
        name_scope = 'Layer_' + str(i)
        net = tf.layers.conv2d(
                inputs = net,
                filters = 32,
                kernel_size = 3,
                padding = 'same',
                activation = f_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer,
                name = name_scope)
        
        if (i % 2) == 1:
            al_loss(net, target_list, target_index, name_scope, train_steps, update_opss, learning_rate, bridge_function, neuron_num)
            net = tf.stop_gradient(net)
            target_index += 1
        
    name_scope = 'Layer_' + str(hidden_layer_count - 1)
    net = tf.layers.conv2d(
            inputs = net,
            filters = 32,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = f_function,
            kernel_initializer = tf.keras.initializers.he_normal(),
            kernel_regularizer = regularizer,
            name = name_scope)

    al_loss(net, target_list, target_index, name_scope, train_steps, update_opss, learning_rate, bridge_function, neuron_num)
    net = tf.stop_gradient(net)
    target_index += 1
    
    for i in range(hidden_layer_count, 2*hidden_layer_count - 1):
        name_scope = 'Layer_' + str(i)
        net = tf.layers.conv2d(
                inputs = net,
                filters = 64,
                kernel_size = 3,
                padding = 'same',
                activation = f_function,
                kernel_initializer = tf.keras.initializers.he_normal(),
                kernel_regularizer = regularizer,
                name = name_scope)
                
        if (i % 2) == 1:
            al_loss(net, target_list, target_index, name_scope, train_steps, update_opss, learning_rate, bridge_function, neuron_num)
            net = tf.stop_gradient(net)
            target_index += 1
        
    name_scope = 'Layer_' + str(2*hidden_layer_count - 1)
    net = tf.layers.conv2d(
            inputs = net,
            filters = 64,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = f_function,
            kernel_initializer = tf.keras.initializers.he_normal(),
            kernel_regularizer = regularizer,
            name = name_scope)

    dense = al_loss(net, target_list, target_index, name_scope, train_steps, update_opss, learning_rate, bridge_function, neuron_num)
    net = tf.stop_gradient(net)
    target_index += 1

    all_group_train_steps = tf.group(train_steps + rev_train_steps + update_opss)

    #inference
    with tf.name_scope('inference'):
        for i in range(num_of_ae):
            dense = y_function(tf.matmul(dense, i_weights[i]) + i_biases[i])
    return dense, all_group_train_steps, global_step, neuron_num

def ResNet_20(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    num_blocks = 3
    regulariztion_scale = 5e-4
    neuron_num = 500
    f_function = tf.nn.relu

    return ResNet(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, 
                    learning_rate, inv_learning_rate, f_function, is_training)

def ResNet_20_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    num_blocks = 3
    regulariztion_scale = 1e-4
    neuron_num = 500
    f_function = tf.nn.elu
    bridge_function = tf.nn.sigmoid
    y_function = tf.nn.sigmoid
    acti_functions = [f_function, y_function, bridge_function]

    return ResNet_AL(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, 
                    learning_rate, inv_learning_rate, acti_functions, is_training)

def ResNet_32(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    num_blocks = 5
    regulariztion_scale = 5e-4
    neuron_num = 500
    f_function = tf.nn.relu

    return ResNet(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, 
                    learning_rate, inv_learning_rate, f_function, is_training)

def ResNet_32_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    num_blocks = 5
    regulariztion_scale = 1e-4
    neuron_num = 500
    f_function = tf.nn.elu
    y_function = tf.nn.sigmoid
    bridge_function = tf.nn.sigmoid
    acti_functions = [f_function, y_function, bridge_function]

    return ResNet_AL(num_blocks, regulariztion_scale, neuron_num, X_placeholder, y_placeholder, 
                    learning_rate, inv_learning_rate, acti_functions, is_training)

def VGG(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    neuron_num = 500
    f_function = tf.nn.relu
    regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4)

    #model
    net = conv_layer_bn(X_placeholder, 128, regularizer, is_training, f_function)
    net = conv_layer_bn(net, 256, regularizer, is_training, f_function)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    
    net = conv_layer_bn(net, 256, regularizer, is_training, f_function)
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    
    net = tf.reshape(net, [-1, 512*2*2])
    
    #addition layer for bridge
    net = tf.layers.dense(net, 5*neuron_num, None, regularizer)
    net = tf.nn.relu(tf.layers.batch_normalization(net, training = is_training))
    net = tf.layers.dense(net, neuron_num, None, regularizer)
    net = tf.nn.relu(tf.layers.batch_normalization(net, training = is_training))
    #end of addition

    for i in range(3):
        net = tf.layers.dense(net, neuron_num, None, regularizer)
        net = f_function(tf.layers.batch_normalization(net, training = is_training))
    net = tf.layers.dense(net, class_num, None, regularizer)
    
    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net, labels = y_placeholder)) + l2_loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_or_create_global_step()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    train_steps = tf.group([train_step, update_ops])
    return net, train_steps, global_step, neuron_num

def VGG_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate, is_training):
    neuron_num = 500
    f_function = tf.nn.elu
    y_function = tf.nn.sigmoid
    bridge_function = tf.nn.sigmoid

    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-5)

    num_of_ae = 4
    train_steps = []
    update_opss = []

    rev_train_steps, global_step, target_list, i_weights, i_biases = \
        autoencoders(y_placeholder, neuron_num, num_of_ae, inv_learning_rate, y_function)

    #model
    name_scope = 'Layer_0' #32x32
    net = conv_layer_bn(X_placeholder, 128, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)

    name_scope = 'Layer_1'
    net = conv_layer_bn(net, 256, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    al_loss(net, target_list, 0, ['Layer_0', 'Layer_1'], train_steps, update_opss, learning_rate, bridge_function, neuron_num, regularizer)
    net = tf.stop_gradient(net)
    
    name_scope = 'Layer_2' #16x16
    net = conv_layer_bn(net, 256, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)

    name_scope = 'Layer_3'
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    al_loss(net, target_list, 1, ['Layer_2', 'Layer_3'], train_steps, update_opss, learning_rate, bridge_function, neuron_num, regularizer)
    net = tf.stop_gradient(net)
    
    name_scope = 'Layer_4' #8x8
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)

    name_scope = 'Layer_5' #4x4
    net = conv_layer_bn(net, 512, regularizer, is_training, f_function, name_scope = name_scope, is_batchnorm = False)
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)
    al_loss(net, target_list, 2, ['Layer_4', 'Layer_5'], train_steps, update_opss, learning_rate, bridge_function, neuron_num, regularizer)
    net = tf.stop_gradient(net)

    name_scope = 'Layer_6' #2x2
    net = tf.reshape(net, [-1, 512*2*2])
    net = tf.layers.dense(net, 1024, tf.nn.elu, regularizer,
            kernel_initializer = tf.keras.initializers.he_normal(), name = name_scope)

    name_scope = 'Layer_7'
    net = tf.layers.dense(net, 1024, tf.nn.elu, regularizer,
            kernel_initializer = tf.keras.initializers.he_normal(), name = name_scope)
    dense = al_loss(net, target_list, 3, ['Layer_6', 'Layer_7'], train_steps, update_opss, learning_rate, bridge_function, neuron_num, regularizer)

    all_group_train_steps = tf.group(train_steps + rev_train_steps + update_opss)

    #inference
    with tf.name_scope('inference'):
        for i in range(num_of_ae):
            dense = y_function(tf.matmul(dense, i_weights[i]) + i_biases[i])
    return dense, all_group_train_steps, global_step, neuron_num