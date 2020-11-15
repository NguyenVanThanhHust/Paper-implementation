from PIL import Image, ImageDraw
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
import cv2

from utils import box_cxcywh_to_xyxy, rescale_bboxes, save_results, plot_results

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
print(bboxes_scaled)

# we want to visulize last decoder layers
# use lists to store the outputs via up-values
conv_features, enc_atten_weights, dec_atten_weights = [], [], []
hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, intput, output: conv_features.append(output)
    ), 
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, intput, output: enc_atten_weights.append(output[1])
    ), 
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, intput, output: dec_atten_weights.append(output[1])
    ),
]

# propagate through the model
outputs = model(img)
for hook in hooks:
    hook.remove()

conv_features = conv_features[0]
enc_atten_weights = enc_atten_weights[0]
dec_atten_weights = dec_atten_weights[0]

# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]
print(h, w)
for idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes_scaled):
    # create rectangle image 
    temp_im = im
    draw_img = ImageDraw.Draw(temp_im)
    image = dec_atten_weights[0, idx].view(h, w)
    trans = T.ToPILImage()
    image = trans(image).convert("RGB")
    image = image.save("attention_map_" + str(idx) + ".jpg")
    shape = [(xmin, ymin), (xmax, ymax)]
    draw_img.rectangle(shape, fill = (255,0,0))
    temp_im.save("object_bbox" + str(idx) + ".jpg")

# output of the CNN
f_map = conv_features['0']
print("Encoder attention:      ", enc_atten_weights[0].shape)
print("Feature map:            ", f_map.tensors.shape)

# get the HxW shape of the feature maps of the CNN
shape = f_map.tensors.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_atten_weights[0].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)
