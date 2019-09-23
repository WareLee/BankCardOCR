from keras.models import Model
from keras.layers import Lambda, Dense, Bidirectional, GRU, Flatten, TimeDistributed, Permute, Activation, Input, Add
from keras.layers import LSTM, Reshape, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.regularizers import l2
from keras.utils import plot_model
from keras.initializers import he_normal
from functools import reduce
from functools import wraps

from keras.layers.advanced_activations import LeakyReLU
from trainning.utils import ctc_loss_layer


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_27_blstm_ctc(is_training=True, img_shape=(32, 256, 1), num_classes=11, max_label_length=26):
    '''darknet53 Darknent body having 52 Convolution2D layers'''
    initializer = he_normal()
    picture_height, picture_width, picture_channel = img_shape

    inputs = Input(shape=(picture_height, picture_width, picture_channel), name='pic_inputs')  # H×W×1 32*256*1
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)  # 32x256x32
    x = resblock_body(x, 64, 1)  # 16x128x64
    x = resblock_body(x, 128, 2)  # 8x64x128
    # x = resblock_body(x, 256, 8)
    x = ZeroPadding2D(((1, 0), (33, 32)))(x)
    x = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(x)
    for i in range(8):
        y = compose(
            DarknetConv2D_BN_Leaky(256 // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(256, (3, 3)))(x)
        x = Add()([x, y])
    # x = resblock_body(x, 512, 8)
    x = ZeroPadding2D(((1, 0), (33, 32)))(x)
    x = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(x)
    for i in range(8):
        y = compose(
            DarknetConv2D_BN_Leaky(512 // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(512, (3, 3)))(x)
        x = Add()([x, y])
    # x = resblock_body(x, 1024, 4)
    x = ZeroPadding2D(((1, 0), (33, 32)))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2))(x)
    for i in range(4):
        y = compose(
            DarknetConv2D_BN_Leaky(1024 // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(1024, (3, 3)))(x)
        x = Add()([x, y])
    # 1x64x1024

    # Map2Sequence part
    x = Permute((2, 3, 1), name='permute')(x)  # 64*1024*1
    rnn_input = TimeDistributed(Flatten(), name='for_flatten_by_time')(x)  # 64*1024

    # RNN part
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), merge_mode='sum',
                      name='LSTM_1')(rnn_input)  # 64*512
    y = BatchNormalization(name='BN_8')(y)
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)  # 64*512

    # 尝试跳过rnn层
    y_pred = Dense(num_classes, activation='softmax', name='y_pred')(y)  # 64*11 这用来做evaluation 和 之后的test检测
    # 在backend的实现ctc_loss的时候没有执行softmax操作所以这里必须要在使用softmax!!!!
    base_model = Model(inputs=inputs, outputs=y_pred)
    print('BASE_MODEL: ')
    base_model.summary()

    # Transcription part (CTC_loss part)
    y_true = Input(shape=[max_label_length], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')

    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [y_true, y_pred, y_pred_length, y_true_length])

    model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print("FULL_MODEL: ")
    model.summary()
    if is_training:
        return model
    else:
        return base_model


def darknet_5():
    """5*conv"""
    inputs = Input(shape=(32, 256, 3))
    x1 = compose(
        DarknetConv2D_BN_Leaky(64, (3, 3)),  # 16->64
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),  # 32 -> 128
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        DarknetConv2D_BN_Leaky(256, (3, 3)),  # 64 -> 256
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),  # 128->512
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        DarknetConv2D_BN_Leaky(512, (3, 3)))(inputs)  # 256->512
    x2 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(x1)

    darknet5 = Model(inputs, x2)  # output: 1x64x512
    darknet5.summary()
    plot_model(darknet5, 'darknet_5.png', show_shapes=True)


def darknet_7_blstm_ctc(is_training=True, img_shape=(32, 256, 1), num_classes=11, max_label_length=26):
    """7*conv"""
    initializer = he_normal()
    picture_height, picture_width, picture_channel = img_shape

    inputs = Input(shape=(picture_height, picture_width, picture_channel), name='pic_inputs')  # H×W×1 32*256*1
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),  # 32, 256, 16
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),  # 32, 256, 32
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),  # 16, 128, 32
        DarknetConv2D_BN_Leaky(64, (3, 3)),  # 16, 128, 64
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),  # 8, 64, 64
        DarknetConv2D_BN_Leaky(128, (3, 3)),  # 8, 64, 128
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),  # 4, 64, 128
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)  # 4, 64, 256
    x2 = compose(
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),  # 2, 64, 256
        DarknetConv2D_BN_Leaky(512, (3, 3)),  # 2, 64, 512
        MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),  # 1, 64, 512
        DarknetConv2D_BN_Leaky(512, (1, 1)))(x1)  # 1, 64, 512

    # Map2Sequence part
    x = Permute((2, 3, 1), name='permute')(x2)  # 64*512*1
    rnn_input = TimeDistributed(Flatten(), name='for_flatten_by_time')(x)  # 64*512

    # RNN part
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), merge_mode='sum',
                      name='LSTM_1')(rnn_input)  # 64*512
    y = BatchNormalization(name='BN_8')(y)
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)  # 64*512

    # 尝试跳过rnn层
    y_pred = Dense(num_classes, activation='softmax', name='y_pred')(y)  # 64*11 这用来做evaluation 和 之后的test检测
    # 在backend的实现ctc_loss的时候没有执行softmax操作所以这里必须要在使用softmax!!!!
    base_model = Model(inputs=inputs, outputs=y_pred)
    print('BASE_MODEL: ')
    base_model.summary()

    # Transcription part (CTC_loss part)
    y_true = Input(shape=[max_label_length], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')

    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [y_true, y_pred, y_pred_length, y_true_length])

    model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    print("FULL_MODEL: ")
    model.summary()
    if is_training:
        return model
    else:
        return base_model


if __name__ == '__main__':
    # darknet_5()  # Trainable params: 6,557,072
    darknet_7_blstm_ctc(is_training=True)
    # darknet_27_blstm_ctc(is_training=True)
