#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
CLASSES_NAME = (
        # "__background__ ",
    'airplane','airport','baseballfield','basketballcourt','bridge',
    'chimney','dam','expressway-service-area','expressway-toll-station',
    'golffield','groundtrackfield','harbor','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill')


def draw_CAM(features,classifier_result,img_path,save_path):
    ''' para:
    features: The tensor  to be visualize [1,C,H,W]
    classifier_result : The precited tensor result for classification after sigmoid or softmax [0.5,0.1,0.2,...]
    img_path: The image path to be load for draw 
    save_path: The heatmap path to be saved 

    '''

    features=features
    output=classifier_result

    # 预测得分最高的那一类对应的输出score / 指定某一类 pred=10
    pred = torch.argmax(output).item()
    
    pred_class = output[:, pred]

    y_grad = []
    def grad_hook(grad):
        y_grad.append(grad)
        

    features.register_hook(grad_hook)
    
    pred_class.backward() # 计算梯度
    
    grads = y_grad[0]   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]

    # 最后一层feature的通道数
    for i in range(features.shape[1]):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap  # 这里的0.4是热力图强度因子
    # print(superimposed_img)
    cv2.imwrite('./work/{}.jpg'.format(CLASSES_NAME[pred]+str(round(output[0].detach().cpu().numpy()[pred],4))), superimposed_img)  # 将图像保存到硬盘


