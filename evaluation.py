"""
评估在测试集上的准确度Accuracy
@WareLee
"""
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from trainning.vgg_blstm_ctc import model
from trainning.darknet_blstm_ctc import darknet_7_blstm_ctc
import cv2
import numpy as np
import keras

char2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3,
                 '4': 4, '5': 5, '6': 6, '7': 7,
                 '8': 8, '9': 9, '_': 10}
num2char_dict = {value: key for key, value in char2num_dict.items()}


def evaluate_generator(val_txt_path, batch_size):
    assert os.path.exists(val_txt_path), 'val_txt does not exist: {}'.format(val_txt_path, )
    batch_start = 0
    with open(val_txt_path) as f:
        lines = f.readlines()
        while batch_start < len(lines):
            batch_end = batch_start + batch_size if batch_start + batch_size <= len(lines) else len(lines)
            lines_batch = lines[batch_start:batch_end]
            batch_start = batch_end
            lines_batch_tmp = [line.strip().split(' ') for line in lines_batch]
            imgpaths_batch = [line[0] for line in lines_batch_tmp]
            labels_batch = [''.join(line[1:]) for line in lines_batch_tmp]
            yield imgpaths_batch, labels_batch


def PredictLabels_by_annofile(model_for_pre, val_txt_path, img_shape, downsample_factor, praent_dir='', batch_size=64,
                              weight_path=None):
    img_h, img_w, img_c = img_shape
    tlt_imgs_evaluated = 0  # 总共被推理的图片数
    correct_evaluated = 0  # 其中被正确推理的图片个数
    tlt_chars = 0  # 推理中涉及的总字符数
    correct_eval_chars = 0  # 其中被正确推理的字符个数

    gt_vs_eval_file = 'gt_vs_eval.txt'
    gt_vs_eval_f = open(gt_vs_eval_file, 'w')  # 保存推理结果

    if weight_path is not None:  # 表明传入的是一个空壳，需要加载权重参数
        print('Loading model : {}'.format(weight_path, ))
        model_for_pre.load_weights(weight_path, by_name=True)  # by_name = True 表示按名字，只取前面一部分的权重
    print("Predicting Start!")

    # 数据准备
    evaluate_gen = evaluate_generator(val_txt_path, batch_size)
    counter = 1
    for imgpaths_batch, labels_batch in evaluate_gen:
        l_ipb = len(imgpaths_batch)
        img_batch = np.zeros((l_ipb, img_h, img_w, img_c))
        print("Eval: {} imgs evaluated ... ".format(counter * batch_size))
        counter += 1
        for i, img_path in enumerate(imgpaths_batch):
            img = cv2.imread(os.path.join(praent_dir, img_path))
            if img_c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_w, img_h))
            if img_c == 1:
                img = np.expand_dims(img, axis=-1)
            # img = img / 255.0 * 2.0 - 1.0  # 零中心化
            img = img / 255.0
            img_batch[i, :, :, :] = img

        # 传输进base_net获得预测的softmax后验概率矩阵
        y_pred_probMatrix = model_for_pre.predict(img_batch)
        y_pred_length = np.full((l_ipb,), int(img_w // downsample_factor))

        # Decode 阶段
        y_pred_labels_tensor_list, _ = keras.backend.ctc_decode(y_pred_probMatrix, y_pred_length,
                                                                greedy=True)  # 使用的是最简单的贪婪算法
        y_pred_labels_tensor = y_pred_labels_tensor_list[0]
        y_pred_labels = keras.backend.get_value(y_pred_labels_tensor)  # 现在还是字符编码
        # 转换成字符串
        y_pred_text = ["" for _ in range(l_ipb)]
        for k in range(l_ipb):
            label = y_pred_labels[k]
            for num in label:
                if num == -1: break
                y_pred_text[k] += num2char_dict[num]

        tlt_imgs_evaluated += l_ipb

        for imgp, label_eval, label_true in zip(imgpaths_batch, y_pred_text, labels_batch):
            correct_evaluated += (label_eval == label_true)
            tlt_chars += len(label_true)
            length = len(label_true) if len(label_true) <= len(label_eval) else len(label_eval)
            for index in range(length):
                correct_eval_chars += (label_true[index] == label_eval[index])

            gt_vs_eval_f.write(' '.join([imgp, label_true, label_eval]) + '\n')

    gt_vs_eval_f.close()
    print('Totle imgs : {}'.format(tlt_imgs_evaluated, ))
    print('PTrue imgs : {}'.format(correct_evaluated, ))
    print('Sample-Level Accuracy: {}'.format(correct_evaluated / tlt_imgs_evaluated))
    print('Char-Level Accuracy: {}'.format(correct_eval_chars / tlt_chars))
    print("Evaluation Finished!")


def _main_vgg_7conv():
    img_shape = (32, 256, 3)
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    batch_size = 64

    # 用于测试的数据
    praent_dir = os.path.abspath('./trainning')
    val_txt_path = os.path.abspath("./trainning/val.txt")

    # 加载模型
    # Totle imgs: 8796   PTrue imgs: 5173    Sample-Accuracy: 0.5881082310140974  Char-Level Accuracy: 0.955670402861673
    # weight_path = './models/train_weight.h5'  # channel==1

    # Totle imgs: 8796   PTrue imgs: 4858  Sample-Accuracy: 0.5522964984083675  Char-Level Accuracy: 0.9424263870829699
    # weight_path = './trainning/models/model1/ep044-loss2.693-val_loss2.419.h5'  # channel==1

    # Totle imgs: 8796  PTrue  imgs: 10  Sample-Accuracy: 0.001136880400181901  Char-Level Accuracy: 0.21360
    weight_path = './trainning/models/model2/ep042-loss2.671-val_loss2.392.h5' # channel==3

    pre_model = model(is_training=False, img_shape=img_shape, num_classes=num_classes, max_label_length=26)
    PredictLabels_by_annofile(pre_model, val_txt_path, img_shape, downsample_factor, praent_dir=praent_dir,
                              batch_size=batch_size, weight_path=weight_path)

def _main_darknet_7conv():
    img_shape = (32, 256, 1)
    num_classes = 11  # 包含“blank”
    max_label_length = 26
    downsample_factor = 4
    batch_size = 64

    # 用于测试的数据
    praent_dir = os.path.abspath('./trainning')
    val_txt_path = os.path.abspath("./trainning/val.txt")

    # 加载模型
    # Sample - Level Accuracy: 0.560368349249659
    # Char - Level Accuracy: 0.9421630429415541
    weight_path = './trainning/models/model4/ep032-loss2.778-val_loss2.538.h5'  # channel==1

    pre_model = darknet_7_blstm_ctc(is_training=False, img_shape=img_shape, num_classes=num_classes, max_label_length=26)
    PredictLabels_by_annofile(pre_model, val_txt_path, img_shape, downsample_factor, praent_dir=praent_dir,
                              batch_size=batch_size, weight_path=weight_path)



if __name__ == '__main__':
    _main_vgg_7conv()
    # _main_darknet_7conv()
