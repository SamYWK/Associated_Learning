import tensorflow as tf
import os

class_num = 10

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

def MLP(X_placeholder, y_placeholder, learning_rate, inv_learning_rate):
    f_function = tf.nn.relu
    neuron_num = 1024

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

def MLP_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate):
    f_function = tf.nn.elu
    bridge_function = tf.nn.sigmoid
    y_function = tf.nn.sigmoid
    neuron_num = 1024

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
    
def CNN(X_placeholder, y_placeholder, learning_rate, inv_learning_rate):
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

def CNN_AL(X_placeholder, y_placeholder, learning_rate, inv_learning_rate):
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