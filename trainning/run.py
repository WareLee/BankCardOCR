import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import platform

if platform.platform().startswith('Linux'):
    import sys

    sys.path.append(os.path.abspath('..'))
from trainning.train import train_model
from trainning import vgg_blstm_ctc
from trainning import darknet_blstm_ctc
from trainning import vgg16_blstm_ctc


def vgg_11_main():
    # 各种路径 以及参数
    weight_save_path = "models/vgg_7_c1/"
    # 数字训练路径

    train_txt_path = "train.txt"
    val_txt_path = "val.txt"
    img_shape = (32, 256, 1)

    # 各种训练时候的参数
    batch_size = 64
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    epochs = 100
    # vgg_7conv_based
    model_for_train = vgg_blstm_ctc.model(is_training=True, img_shape=img_shape, num_classes=num_classes,
                                          max_label_length=max_label_length)

    # 训练模型
    train_model(model_for_train, train_txt_path, val_txt_path,
                weight_save_path, epochs=epochs, img_shape=img_shape, batch_size=batch_size,
                max_label_length=max_label_length, down_sample_factor=downsample_factor)

    return 0


def darknet_7_main():
    # 各种路径 以及参数
    weight_save_path = "models/model4/"
    # 数字训练路径

    train_txt_path = "train.txt"
    val_txt_path = "val.txt"
    img_shape = (32, 256, 1)

    # 各种训练时候的参数
    batch_size = 64
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    epochs = 100
    # vgg_7conv_based
    # model_for_train = vgg_blstm_ctc.model(is_training=True, img_shape=img_shape, num_classes=num_classes,
    #                                       max_label_length=max_label_length)
    # darknet_27conv based
    # model_for_train = darknet_blstm_ctc.darknet_27_blstm_ctc(is_training=True, img_shape=img_shape, num_classes=num_classes,
    #                                       max_label_length=max_label_length)
    # darknet_27conv based
    model_for_train = darknet_blstm_ctc.darknet_7_blstm_ctc(is_training=True, img_shape=img_shape,
                                                            num_classes=num_classes,
                                                            max_label_length=max_label_length)
    # 训练模型
    train_model(model_for_train, train_txt_path, val_txt_path,
                weight_save_path, epochs=epochs, img_shape=img_shape, batch_size=batch_size,
                max_label_length=max_label_length, down_sample_factor=downsample_factor)

    return 0


def vgg_16_org_main():
    # 各种路径 以及参数
    weight_save_path = "models/vgg_16_org_c1/"
    # 数字训练路径

    train_txt_path = "train.txt"
    val_txt_path = "val.txt"
    img_shape = (32, 256, 1)

    # 各种训练时候的参数
    batch_size = 64
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    epochs = 100
    model_for_train = vgg16_blstm_ctc.vgg16_org(is_training=True, img_shape=img_shape,
                                                num_classes=num_classes,
                                                max_label_length=max_label_length)
    # 训练模型
    train_model(model_for_train, train_txt_path, val_txt_path,
                weight_save_path, epochs=epochs, img_shape=img_shape, batch_size=batch_size,
                max_label_length=max_label_length, down_sample_factor=downsample_factor)

    return 0


def vgg_16_BN_main():
    # 各种路径 以及参数
    weight_save_path = "models/vgg_16_BN_c1/"
    # 数字训练路径

    train_txt_path = "train.txt"
    val_txt_path = "val.txt"
    img_shape = (32, 256, 1)

    # 各种训练时候的参数
    batch_size = 64
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    epochs = 100
    model_for_train = vgg16_blstm_ctc.vgg16_BN(is_training=True, img_shape=img_shape,
                                               num_classes=num_classes,
                                               max_label_length=max_label_length)
    # 训练模型
    train_model(model_for_train, train_txt_path, val_txt_path,
                weight_save_path, epochs=epochs, img_shape=img_shape, batch_size=batch_size,
                max_label_length=max_label_length, down_sample_factor=downsample_factor)

    return 0


if __name__ == "__main__":
    # darknet_7_main()
    # vgg_11_main()
    vgg_16_org_main()
