from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
import cv2

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    # plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        print(xmin, ymin, xmax, ymax)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    
def save_results(cv_img, prob, boxes):
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        color = color = (255, 0, 0) 
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1

        cv_img = cv2.rectangle(cv_img, start_point, end_point, color, thickness) 

        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        text_point = (int(xmin), int(ymin) + 50)
        cv_img = cv2.putText(cv_img, text, text_point, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        # print(text, start_point, end_point)
    cv2.imwrite("cat_result.jpg", cv_img)