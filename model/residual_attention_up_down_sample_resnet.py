import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import cv2

################################################### Final Residual Attention ############################################
def compute_saliency(f_maps, mode = "avg"):

    # if mode == "avg":
    #     f_maps = tf.nn.relu(f_maps)
    #     s_map = tf.reduce_mean(f_maps, axis = -1, keepdims = True)
    # elif mode == "max":
    #     f_maps = tf.nn.relu(f_maps)
    #     s_map = tf.reduce_max(f_maps, axis = -1, keepdims = True)
    # elif mode == "sum_abs":
    #     f_maps = tf.abs(f_maps)
    #     s_map = tf.reduce_sum(f_maps, axis = -1, keepdims = True)
    #
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



def balanced_data(x_train, y_train):
    # 0 normal, 1 abnormal
    x_cache = []
    y_cache = []
    count_abnormal = 0
    count_normal = 0
    for k in range(y_train.shape[0]):
        if y_train[k] == 1:
            count_abnormal += 1
            for _ in range(4):
                x_cache.append(x_train[k])
                y_cache.append(y_train[k])
        else:
            count_normal += 1
            x_cache.append(x_train[k])
            y_cache.append(y_train[k])

    x_cache = np.array(x_cache)
    y_cache = np.array(y_cache)
    # print('count_abnormal:', count_abnormal)
    # print('count_normal:', count_normal)

    return x_cache, y_cache


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
    if X.shape[-1] != (out_channels // 4) or stride != 1:

        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=-1, training=training)
        X_shortcut = tf.nn.relu(X_shortcut)
        X_shortcut = tf.layers.conv2d(X_shortcut, filters=out_channels, kernel_size=(1, 1), strides=(stride, stride), padding='same',
                                      kernel_initializer=tf.initializers.glorot_normal())

    # Final step: Add Branch1 and Branch2
    X = X + X_shortcut

    return X


def Trunk_block(X, out_channels, training):

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)
    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    return X


def interpolation(input_tensor, ref_tensor):  # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])


def Attention_1(X, out_channels, training):


    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X_Trunk = Trunk_block(X, out_channels = out_channels,  training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    Residual_id_3_Down_shortcut = X

    Residual_id_3_Down_branched = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    Residual_id_2_Down_shortcut = X

    Residual_id_2_Down_branched = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = interpolation(X, ref_tensor=Residual_id_2_Down_shortcut)

    X = X + Residual_id_2_Down_branched

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = interpolation(X, ref_tensor=Residual_id_3_Down_shortcut)

    X = X + Residual_id_3_Down_branched

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = interpolation(X, ref_tensor=X_Trunk)

    X = tf.layers.batch_normalization(X, axis=-1, training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, out_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X = tf.layers.batch_normalization(X, axis=-1,  training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, 1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X_mask = tf.nn.sigmoid(X)

    X = tf.multiply(X_Trunk, X_mask)

    X = tf.add(X_Trunk, X)

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    return X, compute_saliency(X_mask)

def Attention_2(X, out_channels, training):

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X_Trunk = Trunk_block(X, out_channels = out_channels,  training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    Residual_id_2_Down_shortcut = X

    Residual_id_2_Down_branched = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)


    X = interpolation(X, ref_tensor=Residual_id_2_Down_shortcut)

    X = X + Residual_id_2_Down_branched

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = interpolation(X, ref_tensor=X_Trunk)

    X = tf.layers.batch_normalization(X, axis=-1, training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, out_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X = tf.layers.batch_normalization(X, axis=-1,  training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, 1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X_mask = tf.nn.sigmoid(X)

    X = tf.multiply(X_Trunk, X_mask)

    X = X_Trunk + X

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    return X, compute_saliency(X_mask)

def Attention_3(X, out_channels, training):


    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X_Trunk = Trunk_block(X, out_channels=out_channels, training=training)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    X = interpolation(X, ref_tensor=X_Trunk)

    X = tf.layers.batch_normalization(X, axis=-1, training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, out_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X = tf.layers.batch_normalization(X, axis=-1, training=training)

    X = tf.nn.relu(X)

    X = tf.layers.conv2d(X, 1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer = tf.initializers.glorot_normal())

    X_mask = tf.nn.sigmoid(X)

    X = tf.multiply(X_Trunk, X_mask)

    X = X_Trunk + X

    X = residual_block(X, out_channels = out_channels, stride = 1, training=training)

    return X, compute_saliency(X_mask)




if __name__ == '__main__':

    tf.reset_default_graph()
    data = np.load('heartsound_preprocessed_cnn_exclusive_5s.npz')
    x_train = data['features_train'] / 255.
    y_train = data['labels_train']


    x_test = data['features_test']/ 255.
    y_test = data['labels_test']


    train_split_count = list(data['train_split_count'])
    test_split_count = list(data['test_split_count'])

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)

    input_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    input_y = tf.placeholder(tf.float32, [None, 2])
    training_flag = tf.placeholder(tf.bool)

    X = tf.layers.conv2d(input_x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    X = tf.layers.batch_normalization(X, axis=-1, training = training_flag)
    X = tf.nn.relu(X)

    X = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')
    X = residual_block(X, out_channels = 256, stride = 1, training = training_flag)
    X, X_mask = Attention_1(X, 64, training=training_flag)
    X = residual_block(X, out_channels = 512, stride = 2, training = training_flag)
    X, X_mask2 = Attention_2(X, 512, training=training_flag)
    X = residual_block(X, out_channels = 1024, stride = 2, training = training_flag)
    X, X_mask3 = Attention_3(X, 1024, training=training_flag)
    X = residual_block(X, out_channels = 2048, stride = 2, training = training_flag)
    X = residual_block(X, out_channels = 2048, stride = 1, training = training_flag)
    X = residual_block(X, out_channels = 2048, stride = 1, training = training_flag)

    X = tf.layers.batch_normalization(X, axis=-1, training = training_flag)
    X = tf.nn.relu(X)
    X = tf.layers.average_pooling2d(X, pool_size=(X.shape[1], X.shape[2]), strides=(1, 1))
    X = tf.layers.flatten(X)
    logits = tf.layers.dense(X, units=2, kernel_initializer = tf.initializers.glorot_uniform())
    pred = tf.nn.softmax(logits)
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

    sess = tf.Session()
    init = tf.global_variables_initializer()

    # saver = tf.train.Saver()
    # checkpoint = tf.train.get_checkpoint_state("saved_networks/modified")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

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

        # for batch_num in range(total_batch_train):
        #     x, y = get_batch(x_train, y_train, batch_num, batch_size)
        #     (_, train_loss, train_pred) = sess.run((train, loss, pred), feed_dict={
        #         input_x: x,
        #         input_y: y,
        #         training_flag: True
        #     })
        #     loss_batch_list.append(train_loss)
        #     train_predict.append(train_pred)
        #     train_GT.append(y)
        #
        # train_loss_draw_list.append(np.mean(np.array(loss_batch_list)))
        # print('train_loss:', train_loss_draw_list[-1])
        #
        # train_predict = np.vstack(train_predict)
        # train_GT = np.vstack(train_GT)
        # train_acc = np.mean(np.equal(np.argmax(train_predict, axis = 1), np.argmax(train_GT, axis =  1)).astype(np.float32))
        #
        # print("train_acc:", train_acc)


        test_predict = []
        test_GT = []
        for i in range(total_batch_test):
            x, y = get_batch(x_test, y_test, i, batch_size = batch_size)
            test_pred, s_map1, s_map2, s_map3 = sess.run([pred, X_mask, X_mask2, X_mask3], feed_dict={
                input_x: x,
                training_flag: False
            })

            # try: # the last batch will out of range, so use try-exept
            #     for index in range(batch_size):
            #         img_index = i * batch_size + index
            #         save_img(s_map1[index], img_index, saliency_path, img_name='_att1.jpg', mode = "heatmap")
            #         save_img(s_map2[index], img_index, saliency_path, img_name='_att2.jpg', mode = "heatmap")
            #         save_img(s_map3[index], img_index, saliency_path, img_name='_att3.jpg', mode = "heatmap")
            #         save_img(x_test[index], img_index, saliency_path, img_name = '_input1.jpg', mode = "image")
            #
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
