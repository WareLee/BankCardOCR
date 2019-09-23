# from keras.applications import vgg16
# from keras.utils import plot_model
# model = vgg16.VGG16(include_top=False,weights=None)
# print(model.summary())
# plot_model(model,'vgg16.png',show_shapes=True)

from keras import layers
from keras import models
from keras.utils import plot_model
from keras.initializers import he_normal
from trainning.utils import ctc_loss_layer


def vgg16_BN(is_training=True, img_shape=(32, 256, 1), num_classes=11, max_label_length=26):
    """加了BN版本的vgg16
    Total params: 9,747,136
    Trainable params: 9,739,712
    Non-trainable params: 7,424
    """
    initializer = he_normal()
    picture_height, picture_width, picture_channel = img_shape

    # CNN part  vgg 7*conv
    inputs = layers.Input(shape=(picture_height, picture_width, picture_channel), name='pic_inputs')  # H×W×1 32*256*1
    # conv * 2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_1')(inputs)  # 32*256*64
    x = layers.BatchNormalization(name="BN_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_1.1')(x)  # 32*256*64
    x = layers.BatchNormalization(name="BN_1.1")(x)
    x = layers.Activation("relu", name="relu_1.1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='maxpl_1')(x)  # 16*128*64

    # conv *2
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_2')(x)  # 16*128*128
    x = layers.BatchNormalization(name="BN_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_2.1')(x)  # 16*128*128
    x = layers.BatchNormalization(name="BN_2.1")(x)
    x = layers.Activation("relu", name="relu_2.1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='maxpl_2')(x)  # 8*64*128

    # conv *3
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_3')(x)  # 8*64*256
    x = layers.BatchNormalization(name="BN_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_4')(x)  # 8*64*256
    x = layers.BatchNormalization(name="BN_4")(x)
    x = layers.Activation("relu", name="relu_4")(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_4.1')(x)  # 8*64*256
    x = layers.BatchNormalization(name="BN_4.1")(x)
    x = layers.Activation("relu", name="relu_4.1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpl_3')(x)  # 4*64*256

    # conv * 3
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_5')(x)  # 4*64*512
    x = layers.BatchNormalization(axis=-1, name='BN_5')(x)
    x = layers.Activation("relu", name='relu_5')(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_6')(x)  # 4*64*512
    x = layers.BatchNormalization(axis=-1, name='BN_6')(x)
    x = layers.Activation("relu", name='relu_6')(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True,
                      name='conv2d_6.1')(x)  # 4*64*512
    x = layers.BatchNormalization(axis=-1, name='BN_6.1')(x)
    x = layers.Activation("relu", name='relu_6.1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpl_4')(x)  # 2*64*512

    # conv * 2
    x = layers.Conv2D(512, (2, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer=initializer,
                      use_bias=True, name='conv2d_7')(x)  # 2*64*512
    x = layers.BatchNormalization(name="BN_7")(x)
    x = layers.Activation("relu", name="relu_7")(x)
    x = layers.Conv2D(512, (2, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer=initializer,
                      use_bias=True, name='conv2d_7.1')(x)  # 2*64*512
    x = layers.BatchNormalization(name="BN_7.1")(x)
    x = layers.Activation("relu", name="relu_7.1")(x)
    conv_otput = layers.MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)  # 1*64*512

    # Map2Sequence part
    x = layers.Permute((2, 3, 1), name='permute')(conv_otput)  # 64*512*1
    rnn_input = layers.TimeDistributed(layers.Flatten(), name='for_flatten_by_time')(x)  # 64*512

    # RNN part
    y = layers.Bidirectional(layers.LSTM(256, kernel_initializer=initializer, return_sequences=True), merge_mode='sum',
                             name='LSTM_1')(rnn_input)  # 64*512
    y = layers.BatchNormalization(name='BN_8')(y)
    y = layers.Bidirectional(layers.LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(
        y)  # 64*512

    # 尝试跳过rnn层
    y_pred = layers.Dense(num_classes, activation='softmax', name='y_pred')(y)  # 64*11 这用来做evaluation 和 之后的test检测
    # 在backend的实现ctc_loss的时候没有执行softmax操作所以这里必须要在使用softmax!!!!
    base_model = models.Model(inputs=inputs, outputs=y_pred)
    print('BASE_MODEL: ')
    base_model.summary()

    # Transcription part (CTC_loss part)
    y_true = layers.Input(shape=[max_label_length], name='y_true')
    y_pred_length = layers.Input(shape=[1], name='y_pred_length')
    y_true_length = layers.Input(shape=[1], name='y_true_length')

    ctc_loss_output = layers.Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [y_true, y_pred, y_pred_length, y_true_length])

    model = models.Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print("FULL_MODEL: ")
    model.summary()
    if is_training:
        return model
    else:
        return base_model

    return model


def vgg16_org(is_training=True, img_shape=(32, 256, 1), num_classes=11, max_label_length=26):
    """原始版本的vgg16
    Total params: 9,732,288
    Trainable params: 9,732,288
    Non-trainable params: 0
    """
    initializer = he_normal()
    picture_height, picture_width, picture_channel = img_shape

    inputs = layers.Input(shape=(picture_height, picture_width, picture_channel), name='pic_inputs')
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (2, 2),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (2, 2),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    conv_otput = layers.MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)  # 1*64*512

    # Map2Sequence part
    x = layers.Permute((2, 3, 1), name='permute')(conv_otput)  # 64*512*1
    rnn_input = layers.TimeDistributed(layers.Flatten(), name='for_flatten_by_time')(x)  # 64*512

    # RNN part
    y = layers.Bidirectional(layers.LSTM(256, kernel_initializer=initializer, return_sequences=True), merge_mode='sum',
                             name='LSTM_1')(rnn_input)  # 64*512
    y = layers.BatchNormalization(name='BN_8')(y)
    y = layers.Bidirectional(layers.LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(
        y)  # 64*512

    # 尝试跳过rnn层
    y_pred = layers.Dense(num_classes, activation='softmax', name='y_pred')(y)  # 64*11 这用来做evaluation 和 之后的test检测
    # 在backend的实现ctc_loss的时候没有执行softmax操作所以这里必须要在使用softmax!!!!
    base_model = models.Model(inputs=inputs, outputs=y_pred)
    print('BASE_MODEL: ')
    base_model.summary()

    # Transcription part (CTC_loss part)
    y_true = layers.Input(shape=[max_label_length], name='y_true')
    y_pred_length = layers.Input(shape=[1], name='y_pred_length')
    y_true_length = layers.Input(shape=[1], name='y_true_length')

    ctc_loss_output = layers.Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [y_true, y_pred, y_pred_length, y_true_length])

    model = models.Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print("FULL_MODEL: ")
    model.summary()
    if is_training:
        return model
    else:
        return base_model


if __name__ == '__main__':
    img_shape = (32, 256, 1)
    model = vgg16_BN(img_shape)
    # model = vgg16_org(img_shape=img_shape)
