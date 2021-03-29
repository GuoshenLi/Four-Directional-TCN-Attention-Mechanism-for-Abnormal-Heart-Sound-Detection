import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import os
import cv2

def one_hot_transform(audio_label):
    audio_label = np.array(audio_label).reshape(len(audio_label), 1)
    audio_label = OneHotEncoder().fit(audio_label).transform(audio_label).toarray()
    return audio_label

def train_data_label_shuffle(Xtrain_normalize, Ytrain_onehot):
    train_num = np.shape(Xtrain_normalize)[0]
    new_order = np.random.permutation(train_num)

    Xtrain_normalize = Xtrain_normalize[new_order]
    Ytrain_onehot = Ytrain_onehot[new_order]

    return Xtrain_normalize, Ytrain_onehot


def get_batch(Xtrain_normalize, Ytrain_onehot, number, batch_size):
    return Xtrain_normalize[number * batch_size:(number + 1) * batch_size], \
           Ytrain_onehot[number * batch_size:(number + 1) * batch_size]


def performance_evaluation(C_Matrix):

    TP = C_Matrix[1][1]
    FP = C_Matrix[0][1]
    TN = C_Matrix[0][0]
    FN = C_Matrix[1][0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)

    score = (sensitivity + specificity) / 2.

    return sensitivity, specificity, score

def compute_saliency(f_maps):

    s_map_min = tf.reduce_min(f_maps, axis = [1, 2, 3], keepdims = True) # [batch_size, 8, 8, 1]
    s_map_max = tf.reduce_max(f_maps, axis = [1, 2, 3], keepdims = True)
    s_map = tf.div(f_maps - s_map_min + 1e-8, s_map_max - s_map_min + 1e-8) # [batch_size, 8, 8, 1]

    # s_map = tf.div(s_map, s_map_max)

    # s_map = tf.sigmoid(s_map)
    saliency_map = tf.tile(s_map, (1, 1, 1, 3))
    saliency_map_dis = tf.image.resize_images(saliency_map, (128, 128))  # (8, 128, 128, 3)

    return saliency_map_dis


def save_img(img, img_index, root_path, img_name, mode = "image"):
    img = np.uint8(255 * img)
    if mode == "image":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == "heatmap":
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_path = os.path.join(root_path, str(img_index) + img_name)
    cv2.imwrite(img_path, img)


def temporal_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_prob ,training):

    shortcut = x
    # block1
    x = tf.layers.conv1d(x, filters=nb_filters, kernel_size=kernel_size,dilation_rate=dilation_rate, padding=padding)
    #x = tf.layers.batch_normalization(x, axis=-1, training = training)
    #x = tf.contrib.layers.layer_norm(x)
    x = tf.layers.batch_normalization(x, axis = -1,training = training)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate = dropout_prob, training = training)


    # block2
    x = tf.layers.conv1d(x, filters=nb_filters, kernel_size=kernel_size,dilation_rate=dilation_rate, padding=padding)
    #x = tf.layers.batch_normalization(x, axis=-1, training = training)
    #x = tf.contrib.layers.layer_norm(x)
    x = tf.layers.batch_normalization(x, axis = -1, training = training)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate = dropout_prob, training = training)

    if shortcut.shape[-1] != x.shape[-1]:  # match the dimention
        shortcut = tf.layers.conv1d(shortcut, filters=nb_filters, kernel_size=1, padding='same')

    x = tf.nn.relu(x + shortcut)

    return x


def temporal_conv_net(x, num_channels, kernel_size, dropout_prob, training):
        # num_channels is a list contains hidden sizes of Conv1D
        # The model contains "num_levels" TemporalBlock
    num_levels = len(num_channels)
    for i in range(num_levels):
        dilation_rate = 2 ** i  # exponential growth
        x = temporal_block(x, dilation_rate, num_channels[i], kernel_size, padding='causal', dropout_prob = dropout_prob, training = training)

    return x




def residual_block(X, out_channels, stride, training):
    ##### Branch1 is the main path and Branch2 is the shortcut path #####

    X_shortcut = X

    ##### Branch1 #####
    # First component of Branch1
    X = tf.layers.batch_normalization(X, axis=-1, training=training)
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(X, filters=out_channels // 4, kernel_size=(1, 1), strides=(1, 1), padding='same',
                         kernel_initializer=tf.initializers.glorot_normal())

    # Second component of Branch1
    X = tf.layers.batch_normalization(X, axis=-1, training=training)
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(X, filters=out_channels // 4, kernel_size=(3, 3), strides=(stride, stride), padding='same',
                         kernel_initializer=tf.initializers.glorot_normal())


    # Third component of Branch1
    X = tf.layers.batch_normalization(X, axis=-1, training=training)
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(X, filters=out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
                         kernel_initializer=tf.initializers.glorot_normal())


    ##### Branch2 ####
    if X_shortcut.shape[-1] != out_channels or stride != 1:

        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=-1, training=training)
        X_shortcut = tf.nn.relu(X_shortcut)
        X_shortcut = tf.layers.conv2d(X_shortcut, filters=out_channels, kernel_size=(1, 1), strides=(stride, stride), padding='same',
                                      kernel_initializer=tf.initializers.glorot_normal())

    # Final step: Add Branch1 and Branch2
    X = X + X_shortcut

    return X

def Trunk_block(X, out_channels,training):

    X = residual_block(X, out_channels, stride = 2, training = training)
    X = residual_block(X, out_channels, stride = 1, training = training)

    return X


def interpolation(input_tensor, ref_tensor, name):  # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value], name=name)


def Attention_1(X, out_channels, training):

    patch_size = (2, 2)
    n_hidden = out_channels
    X = residual_block(X, out_channels, stride=1, training=training)
    #print(X)
    X_Trunk = Trunk_block(X, out_channels * 4, training=training)
    #print(X_Trunk)
    cheight, cwidth, cchannels = X.shape[1:]
    pheight, pwidth = patch_size
    psize = int(pheight * pwidth * cchannels)

    # Number of patches in each direction
    npatchesH = int( int(cheight) / pheight )
    npatchesW = int( int(cwidth) / pwidth )

    # Split in patches: bs, #H, ph, #W, pw, cc
    X = tf.reshape(
        X,
        (-1, npatchesH, pheight, npatchesW, pwidth, cchannels))
    #print(X)
    # bs, #H, #W, ph, pw, cc
    X_scan_first = tf.transpose(
        X,
        (0, 1, 3, 2, 4, 5))
    #print(X_scan_first)
    # FIRST SUBLAYER
    # The RNN Layer needs a 3D tensor input: bs*#H, #W, psize
    # bs*#H, #W, ph * pw * cc
    X_scan_first = tf.reshape(
        X_scan_first,
        (-1, npatchesW, psize))

    #print(X_scan_first)

    output_fw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden, n_hidden, n_hidden], kernel_size = 2, dropout_prob=0.3,
                              training=training)
    #print(output_fw_seq)
    X_scan_first = tf.reverse(X_scan_first, axis = [1])
    output_bw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden, n_hidden, n_hidden], kernel_size = 2, dropout_prob=0.3,
                              training=training)
    #print(output_bw_seq)

    output_seq_first = tf.concat([output_fw_seq, output_bw_seq], -1)

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_first = tf.reshape(
        output_seq_first,
        (-1, npatchesH, npatchesW, 2 * n_hidden))
    #print(output_seq_first)
    ######### 2nd direction #######

    # bs, #W, #H, ph, pw, cc
    X_scan_second = tf.transpose(
        X,
        (0, 3, 1, 2, 4, 5))


    # bs * #W, #H, ph*pw*cc
    X_scan_second = tf.reshape(
        X_scan_second,
        (-1, npatchesH, psize))

    #print(X_scan_second)

    output_fw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden, n_hidden, n_hidden], kernel_size = 2, dropout_prob=0.3,
                              training=training)
    #print(output_fw_seq)
    X_scan_second = tf.reverse(X_scan_second, axis = [1])
    output_bw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden, n_hidden, n_hidden], kernel_size = 2, dropout_prob=0.3,
                              training=training)
    #print(output_bw_seq)


    output_seq_second = tf.concat([output_fw_seq, output_bw_seq], -1)
    #print(output_seq_second)

    # Revert reshape: bs, #W, #H, 2*hid
    output_seq_second = tf.reshape(
        output_seq_second,
        (-1, npatchesW, npatchesH, 2 * n_hidden))

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_second = tf.transpose(output_seq_second, (0, 2, 1, 3))
    #print(output_seq_second)
    X = tf.concat([output_seq_first, output_seq_second], axis = -1)
    #print(X)


    left_mask = tf.layers.batch_normalization(X[:,:,:,: out_channels], axis = -1, training = training)
    left_mask = tf.nn.relu(left_mask)
    left_mask = tf.layers.conv2d(left_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    left_mask = tf.nn.sigmoid(left_mask)

    right_mask = tf.layers.batch_normalization(X[:,:,:,out_channels: 2* out_channels], axis = -1, training = training)
    right_mask = tf.nn.relu(right_mask)
    right_mask = tf.layers.conv2d(right_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    right_mask = tf.nn.sigmoid(right_mask)

    down_mask = tf.layers.batch_normalization(X[:,:,:,2 * out_channels: 3* out_channels], axis = -1, training = training)
    down_mask = tf.nn.relu(down_mask)
    down_mask = tf.layers.conv2d(down_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    down_mask = tf.nn.sigmoid(down_mask)

    up_mask = tf.layers.batch_normalization(X[:,:,:,3*out_channels:], axis = -1, training = training)
    up_mask = tf.nn.relu(up_mask)
    up_mask = tf.layers.conv2d(up_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    up_mask = tf.nn.sigmoid(up_mask)



    X_left = tf.multiply(X_Trunk[:,:,:,:out_channels], left_mask)
    X_right = tf.multiply(X_Trunk[:,:,:,out_channels:2* out_channels], right_mask)
    X_down = tf.multiply(X_Trunk[:,:,:,2*out_channels:3*out_channels], down_mask)
    X_up = tf.multiply(X_Trunk[:,:,:,3*out_channels:], up_mask)


    X = X_Trunk + tf.concat([X_left, X_right, X_down, X_up], axis = -1)
    X = residual_block(X, out_channels, stride = 1, training = training)


    return X, compute_saliency(left_mask), compute_saliency(right_mask), compute_saliency(down_mask), compute_saliency(up_mask)

def Attention_2(X, out_channels, training):

    patch_size = (2, 2)
    n_hidden = out_channels
    X = residual_block(X, out_channels, stride=1, training=training)
    # print(X)
    X_Trunk = Trunk_block(X, out_channels * 4, training=training)
    # print(X_Trunk)
    cheight, cwidth, cchannels = X.shape[1:]
    pheight, pwidth = patch_size
    psize = int(pheight * pwidth * cchannels)

    # Number of patches in each direction
    npatchesH = int(int(cheight) / pheight)
    npatchesW = int(int(cwidth) / pwidth)

    # Split in patches: bs, #H, ph, #W, pw, cc
    X = tf.reshape(
        X,
        (-1, npatchesH, pheight, npatchesW, pwidth, cchannels))
    # print(X)
    # bs, #H, #W, ph, pw, cc
    X_scan_first = tf.transpose(
        X,
        (0, 1, 3, 2, 4, 5))
    # print(X_scan_first)
    # FIRST SUBLAYER
    # The RNN Layer needs a 3D tensor input: bs*#H, #W, psize
    # bs*#H, #W, ph * pw * cc
    X_scan_first = tf.reshape(
        X_scan_first,
        (-1, npatchesW, psize))

    # print(X_scan_first)

    output_fw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden, n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_fw_seq)
    X_scan_first = tf.reverse(X_scan_first, axis=[1])
    output_bw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden, n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_bw_seq)

    output_seq_first = tf.concat([output_fw_seq, output_bw_seq], -1)

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_first = tf.reshape(
        output_seq_first,
        (-1, npatchesH, npatchesW, 2 * n_hidden))
    # print(output_seq_first)
    ######### 2nd direction #######

    # bs, #W, #H, ph, pw, cc
    X_scan_second = tf.transpose(
        X,
        (0, 3, 1, 2, 4, 5))

    # bs * #W, #H, ph*pw*cc
    X_scan_second = tf.reshape(
        X_scan_second,
        (-1, npatchesH, psize))

    # print(X_scan_second)

    output_fw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden, n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_fw_seq)
    X_scan_second = tf.reverse(X_scan_second, axis=[1])
    output_bw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden, n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_bw_seq)

    output_seq_second = tf.concat([output_fw_seq, output_bw_seq], -1)
    # print(output_seq_second)

    # Revert reshape: bs, #W, #H, 2*hid
    output_seq_second = tf.reshape(
        output_seq_second,
        (-1, npatchesW, npatchesH, 2 * n_hidden))

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_second = tf.transpose(output_seq_second, (0, 2, 1, 3))
    # print(output_seq_second)
    X = tf.concat([output_seq_first, output_seq_second], axis=-1)
    # print(X)

    left_mask = tf.layers.batch_normalization(X[:, :, :, : out_channels], axis=-1, training=training)
    left_mask = tf.nn.relu(left_mask)
    left_mask = tf.layers.conv2d(left_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    left_mask = tf.nn.sigmoid(left_mask)

    right_mask = tf.layers.batch_normalization(X[:, :, :, out_channels: 2 * out_channels], axis=-1, training=training)
    right_mask = tf.nn.relu(right_mask)
    right_mask = tf.layers.conv2d(right_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    right_mask = tf.nn.sigmoid(right_mask)

    down_mask = tf.layers.batch_normalization(X[:, :, :, 2 * out_channels: 3 * out_channels], axis=-1,
                                              training=training)
    down_mask = tf.nn.relu(down_mask)
    down_mask = tf.layers.conv2d(down_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    down_mask = tf.nn.sigmoid(down_mask)

    up_mask = tf.layers.batch_normalization(X[:, :, :, 3 * out_channels:], axis=-1, training=training)
    up_mask = tf.nn.relu(up_mask)
    up_mask = tf.layers.conv2d(up_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    up_mask = tf.nn.sigmoid(up_mask)

    X_left = tf.multiply(X_Trunk[:, :, :, :out_channels], left_mask)
    X_right = tf.multiply(X_Trunk[:, :, :, out_channels:2 * out_channels], right_mask)
    X_down = tf.multiply(X_Trunk[:, :, :, 2 * out_channels:3 * out_channels], down_mask)
    X_up = tf.multiply(X_Trunk[:, :, :, 3 * out_channels:], up_mask)

    X = X_Trunk + tf.concat([X_left, X_right, X_down, X_up], axis=-1)
    X = residual_block(X, out_channels, stride=1, training=training)

    return X, compute_saliency(left_mask), compute_saliency(right_mask), compute_saliency(down_mask), compute_saliency(up_mask)


def Attention_3(X, out_channels, training):
    patch_size = (2, 2)
    n_hidden = out_channels
    X = residual_block(X, out_channels, stride=1, training=training)
    # print(X)
    X_Trunk = Trunk_block(X, out_channels * 4, training=training)
    # print(X_Trunk)
    cheight, cwidth, cchannels = X.shape[1:]
    pheight, pwidth = patch_size
    psize = int(pheight * pwidth * cchannels)

    # Number of patches in each direction
    npatchesH = int(int(cheight) / pheight)
    npatchesW = int(int(cwidth) / pwidth)

    # Split in patches: bs, #H, ph, #W, pw, cc
    X = tf.reshape(
        X,
        (-1, npatchesH, pheight, npatchesW, pwidth, cchannels))
    # print(X)
    # bs, #H, #W, ph, pw, cc
    X_scan_first = tf.transpose(
        X,
        (0, 1, 3, 2, 4, 5))
    # print(X_scan_first)
    # FIRST SUBLAYER
    # The RNN Layer needs a 3D tensor input: bs*#H, #W, psize
    # bs*#H, #W, ph * pw * cc
    X_scan_first = tf.reshape(
        X_scan_first,
        (-1, npatchesW, psize))

    # print(X_scan_first)

    output_fw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_fw_seq)
    X_scan_first = tf.reverse(X_scan_first, axis=[1])
    output_bw_seq = temporal_conv_net(X_scan_first, num_channels=[n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_bw_seq)

    output_seq_first = tf.concat([output_fw_seq, output_bw_seq], -1)

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_first = tf.reshape(
        output_seq_first,
        (-1, npatchesH, npatchesW, 2 * n_hidden))
    # print(output_seq_first)
    ######### 2nd direction #######

    # bs, #W, #H, ph, pw, cc
    X_scan_second = tf.transpose(
        X,
        (0, 3, 1, 2, 4, 5))

    # bs * #W, #H, ph*pw*cc
    X_scan_second = tf.reshape(
        X_scan_second,
        (-1, npatchesH, psize))

    # print(X_scan_second)

    output_fw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_fw_seq)
    X_scan_second = tf.reverse(X_scan_second, axis=[1])
    output_bw_seq = temporal_conv_net(X_scan_second, num_channels=[n_hidden], kernel_size=2,
                                      dropout_prob=0.3,
                                      training=training)
    # print(output_bw_seq)

    output_seq_second = tf.concat([output_fw_seq, output_bw_seq], -1)
    # print(output_seq_second)

    # Revert reshape: bs, #W, #H, 2*hid
    output_seq_second = tf.reshape(
        output_seq_second,
        (-1, npatchesW, npatchesH, 2 * n_hidden))

    # Revert reshape: bs, #H, #W, 2*hid
    output_seq_second = tf.transpose(output_seq_second, (0, 2, 1, 3))
    # print(output_seq_second)
    X = tf.concat([output_seq_first, output_seq_second], axis=-1)
    # print(X)

    left_mask = tf.layers.batch_normalization(X[:, :, :, : out_channels], axis=-1, training=training)
    left_mask = tf.nn.relu(left_mask)
    left_mask = tf.layers.conv2d(left_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    left_mask = tf.nn.sigmoid(left_mask)

    right_mask = tf.layers.batch_normalization(X[:, :, :, out_channels: 2 * out_channels], axis=-1, training=training)
    right_mask = tf.nn.relu(right_mask)
    right_mask = tf.layers.conv2d(right_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    right_mask = tf.nn.sigmoid(right_mask)

    down_mask = tf.layers.batch_normalization(X[:, :, :, 2 * out_channels: 3 * out_channels], axis=-1,
                                              training=training)
    down_mask = tf.nn.relu(down_mask)
    down_mask = tf.layers.conv2d(down_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    down_mask = tf.nn.sigmoid(down_mask)

    up_mask = tf.layers.batch_normalization(X[:, :, :, 3 * out_channels:], axis=-1, training=training)
    up_mask = tf.nn.relu(up_mask)
    up_mask = tf.layers.conv2d(up_mask, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    up_mask = tf.nn.sigmoid(up_mask)

    X_left = tf.multiply(X_Trunk[:, :, :, :out_channels], left_mask)
    X_right = tf.multiply(X_Trunk[:, :, :, out_channels:2 * out_channels], right_mask)
    X_down = tf.multiply(X_Trunk[:, :, :, 2 * out_channels:3 * out_channels], down_mask)
    X_up = tf.multiply(X_Trunk[:, :, :, 3 * out_channels:], up_mask)

    X = X_Trunk + tf.concat([X_left, X_right, X_down, X_up], axis=-1)
    X = residual_block(X, out_channels, stride=1, training=training)

    return X, compute_saliency(left_mask), compute_saliency(right_mask), compute_saliency(down_mask), compute_saliency(
        up_mask)

if __name__ == '__main__':
    #######  4_dir_TCN ##########
    tf.reset_default_graph()
    data = np.load('heartsound_preprocessed_cnn_exclusive_5s.npz')
    x_train = data['features_train'] / 255.
    y_train = data['labels_train']


    x_test = data['features_test']/ 255.
    y_test = data['labels_test']

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)

    input_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    input_y = tf.placeholder(tf.float32, [None, 2])
    training_flag = tf.placeholder(tf.bool)

    X = tf.layers.conv2d(input_x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')
    X = residual_block(X, out_channels = 64, stride = 1, training = training_flag)
    X, X_left_1, X_right_1, X_down_1, X_up_1 = Attention_1(X, 64, training=training_flag)
    X = residual_block(X, out_channels = 128, stride = 1, training = training_flag)
    X, X_left_2, X_right_2, X_down_2, X_up_2 = Attention_2(X, 128, training=training_flag)
    X = residual_block(X, out_channels = 256, stride = 1, training = training_flag)
    X, X_left_3, X_right_3, X_down_3, X_up_3 = Attention_3(X, 256, training=training_flag)
    # X = residual_block(X, out_channels = 512, stride = 2, training = training_flag)
    # X = residual_block(X, out_channels = 2048, stride = 1, training = training_flag)
    # X = residual_block(X, out_channels = 2048, stride = 1, training = training_flag)

    X = tf.layers.batch_normalization(X, axis=-1, training = training_flag)
    X = tf.nn.relu(X)
    X = tf.layers.average_pooling2d(X, pool_size=(X.shape[1], X.shape[2]), strides=(1, 1))
    X = tf.layers.flatten(X)
    logits = tf.layers.dense(X, units=2, kernel_initializer = tf.initializers.glorot_uniform())
    pred = tf.nn.softmax(logits, axis = -1)

    #print(X)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
    loss = tf.reduce_mean(loss)
    loss += 0.0001 * tf.add_n([tf.nn.l2_loss(val) for val in tf.trainable_variables()])
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(input_y, axis=1)), tf.float32))

    print("Trainable parameters:",
          np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(0.0001)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]

    with tf.control_dependencies(update_ops):
        train = optimizer.apply_gradients(capped_gvs)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        #####################training_code#######################
        saliency_path = 'residual_attention_Saliency'
        if not os.path.exists(saliency_path):
            os.makedirs(saliency_path)

        batch_size = 8
        max_epoch = 200
        total_batch_train = int(x_train.shape[0] / batch_size)
        total_batch_test = int(x_test.shape[0] / batch_size)
        sess.run(init)
        test_acc_draw_list = []
        train_loss_draw_list = []
        for epoch in range(max_epoch):
            train_acc_batch_list = []
            test_acc_batch_list = []
            loss_batch_list = []
            train_predict = []
            train_GT = []

            for batch_num in range(total_batch_train):
                x, y = get_batch(x_train, y_train, batch_num, batch_size)
                (_, train_loss, train_pred) = sess.run((train, loss, pred), feed_dict={
                    input_x: x,
                    input_y: y,
                    training_flag: True
                })
                loss_batch_list.append(train_loss)
                train_predict.append(train_pred)
                train_GT.append(y)

            train_loss_draw_list.append(np.mean(np.array(loss_batch_list)))
            print('train_loss:', train_loss_draw_list[-1])

            train_predict = np.vstack(train_predict)
            train_GT = np.vstack(train_GT)
            train_acc = np.mean(np.equal(np.argmax(train_predict, axis = 1), np.argmax(train_GT, axis =  1)).astype(np.float32))

            print("train_acc:", train_acc)

            test_predict = []
            test_GT = []
            for i in range(total_batch_test):
                x, y = get_batch(x_test, y_test, i, batch_size=batch_size)

                fetch = [pred, X_up_1, X_down_1, X_left_1, X_right_1,
                                            X_up_2, X_down_2, X_left_2, X_right_2,
                                            X_up_3, X_down_3, X_left_3, X_right_3]

                result = sess.run(fetch, feed_dict={
                    input_x: x,
                    training_flag: False
                })

                test_pred, X_up_1_, X_down_1_, X_left_1_, X_right_1_,\
                 X_up_2_, X_down_2_, X_left_2_, X_right_2_,\
                 X_up_3_, X_down_3_, X_left_3_, X_right_3_ = result

                # try: # the last batch will out of range, so use try-exept
                #     for index in range(batch_size):
                #         img_index = i * batch_size + index
                #         save_img(x_train[index],   img_index, saliency_path, img_name='_image.jpg',      mode = "image")
                #         save_img(X_up_1_[index],   img_index, saliency_path, img_name='_att1_up.jpg',      mode = "heatmap")
                #         save_img(X_down_1_[index], img_index, saliency_path, img_name='_att1_down.jpg',    mode = "heatmap")
                #         save_img(X_left_1_[index], img_index, saliency_path, img_name='_att1_left.jpg',    mode = "heatmap")
                #         save_img(X_right_1_[index],img_index, saliency_path, img_name = '_att1_right.jpg', mode = "heatmap")
                #         save_img(X_up_2_[index] ,  img_index, saliency_path, img_name='_att2_up.jpg',      mode = "heatmap")
                #         save_img(X_down_2_[index], img_index, saliency_path, img_name='_att2_down.jpg',    mode = "heatmap")
                #         save_img(X_left_2_[index], img_index, saliency_path, img_name='_att2_left.jpg',    mode = "heatmap")
                #         save_img(X_right_2_[index],img_index, saliency_path, img_name = '_att2_right.jpg', mode = "heatmap")
                #         save_img(X_up_3_[index],   img_index, saliency_path, img_name='_att3_up.jpg',      mode = "heatmap")
                #         save_img(X_down_3_[index], img_index, saliency_path, img_name='_att3_down.jpg',    mode = "heatmap")
                #         save_img(X_left_3_[index], img_index, saliency_path, img_name='_att3_left.jpg',    mode = "heatmap")
                #         save_img(X_right_3_[index],img_index, saliency_path, img_name = '_att3_right.jpg', mode = "heatmap")
                #
                # except:
                #     print('save_all_image_and saliency map of one test epoch!')

                test_predict.append(test_pred)
                test_GT.append(y)

            test_predict = np.vstack(test_predict)
            test_GT = np.vstack(test_GT)

            y_pred_ = np.array(np.argmax(test_predict, axis=1))
            y_test_ = np.array(np.argmax(test_GT, axis=1))

            test_acc = np.mean(np.equal(y_pred_, y_test_).astype(np.float32))

            print("test_acc:", test_acc)
            x_train, y_train = train_data_label_shuffle(x_train, y_train)

            C_Matrix = confusion_matrix(y_test_, y_pred_)  # is reversed!

            se, sp, score = performance_evaluation(C_Matrix)

            print("se: ", se)
            print("sp: ", sp)
            print("score: ", score)
            print('\n')
