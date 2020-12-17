import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def draw_features(width, height, x, savename='./heatmap/'):
    tic = time.time()
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        # print(img.shape)
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map

        heatmap = cv2.resize(img, (800, 800))

        cv2.imwrite(savename+str(i)+'heatmap.jpg', heatmap)
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename+'heatmap.jpg', dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))