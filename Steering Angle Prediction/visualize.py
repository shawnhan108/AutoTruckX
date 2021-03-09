from math import tan, pi 
import io

import cv2 
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import select_model, load_weights, preprocess_img
from config import net, inf_img_src, best_ckpt_src, vis_out_src, target_layer_name

def vis_angle_on_img(img, rad):
    """
    Draws the angle on the image.
    """
    H = img.shape[0]
    W = img.shape[1]

    base_coord = (int(W/2), H)
    angled_coord = (int(W / 2 + tan(rad) * H / 4), int(H * 3 / 4))

    img = cv2.line(img, base_coord, angled_coord, (0, 255, 0), thickness=5)
    img = cv2.putText(img, 'Angle: {0} rad'.format(rad), (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
    
    cv2.imshow("Output", img)
    return img

def vis_salient_map(model, orig_image, img, target_layer_name, dpi=500):
    # returns the original image blended with the salient map, as a PIL PNG.

    # forward propagation
    img.requires_grad_()
    scores = None
    img_trans = img
    for name, layer in model.resnet50._modules.items():
        img_trans = layer(img_trans)  
        if name == target_layer_name:
            scores = img_trans 
            break

    # Get max score, then perform backpropagation on max score
    # Saliency is now the gradient with respect to input image now.
    score_max = torch.max(scores)
    score_max.backward()
    saliency, _ = torch.max(img.grad.data.abs(),dim=1)

    # Visualize salient map
    out = saliency[0].detach().cpu().numpy()
    size = (224, 224)
    plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
    plt.figimage(out, cmap=plt.cm.hot)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=dpi)
    buf.seek(0)

    # Combine with original image
    smap = Image.open(buf).resize(orig_image.size).convert("RGBA")
    buf.close()
    orig_image = orig_image.convert("RGBA")
    blended = Image.blend(orig_image, smap, 0.75)

    return blended

if __name__ == '__main__':
    # init model
    init_msg = "(1) Initiating Visualization ... "
    logger, model = select_model(model_name=net, init_msg=init_msg)

    # load model weights
    load_weights(model, best_ckpt_src, logger)
    for param in model.parameters():
        param.requires_grad = False

    # input Image
    orig_image = Image.open(inf_img_src).convert('RGB')
    img = preprocess_img(np.array(orig_image), net)

    # Visualize and save
    out_png = vis_salient_map(model, orig_image, img, target_layer_name=target_layer_name)
    out_png.save(vis_out_src, "PNG")
