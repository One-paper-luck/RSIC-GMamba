import numpy as np
import cv2
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visulize_attention_ratio(img_path, attention_mask, ratio=1, cmap="jet", i=None):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    # plt.savefig('v1.png', dpi=300)

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    # plt.savefig('./v_result/%s.png' %i, bbox_inches='tight',pad_inches=0)
    plt.savefig('./%s.png' %i, bbox_inches='tight',pad_inches=0)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

img_path='/media/dmd/ours/mlw/rs/UCM_Captions/imgs/2008.tif'

# encoder
for i in range(3):
    feature_map=np.load('./l%s.npy' %i)

    # 1. 对通道取平均，得到49x1的向量
    mean_feature_map = np.mean(feature_map, axis=1).reshape(-1, 1)  # 变为49x1

    # 2. 对平均值进行归一化
    scaler = MinMaxScaler()
    normalized_mean = scaler.fit_transform(mean_feature_map)

    # attention_mask=feature_map.mean(axis=1)
    attention_mask=np.reshape(normalized_mean,(7,7))
    visulize_attention_ratio(img_path,attention_mask, i=i)

