import keras
import cv2
import numpy as np
import sys
from vgg_blstm_ctc import model

char2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3,
                 '4': 4, '5': 5, '6': 6, '7': 7,
                 '8': 8, '9': 9, '_': 10}
num2char_dict = {value: key for key, value in char2num_dict.items()}


def single_recognition(img, model_dir):
    img_w = 256
    img_h = 32
    img_c = 1
    downsample_factor = 4
    if img_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_w, img_h))
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0 * 2.0 - 1.0

    img_batch = np.zeros((1, img_h, img_w, img_c))
    img_batch[0, :, :, :] = img

    # print(gray_img.shape)

    model_for_predict = model(is_training=False, img_shape=(img_h, img_w, img_c), num_classes=11, max_label_length=26)
    # model_for_predict = vgg_b_ctc.model(is_training=False, img_size=(256,32), num_classes=11, max_label_length=25)
    model_for_predict.load_weights(model_dir)

    y_pred_probMatrix = model_for_predict.predict(img_batch)
    # Decode 阶段
    y_pred_labels_tensor_list, _ = keras.backend.ctc_decode(y_pred_probMatrix, [img_w // downsample_factor],
                                                            greedy=True)  # 使用的是最简单的贪婪算法
    y_pred_labels_tensor = y_pred_labels_tensor_list[0]
    y_pred_labels = keras.backend.get_value(y_pred_labels_tensor)  # 现在还是字符编码
    # 转换成字符串
    y_pred_text = ''
    for num in y_pred_labels[0]:
        y_pred_text += num2char_dict[num]
    # print(y_pred_labels)
    return y_pred_text
