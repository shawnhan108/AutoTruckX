import io

import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from utils import select_model, load_weights, preprocess_img
from config import best_ckpt_src, net, inf_img_src, inf_out_img_src, IMG_DIM, device, \
                    inf_vid_src, inf_out_vid_src

def inference_image(model, logger, img=np.array(Image.open(inf_img_src)), record=True, dpi=500):

    img = preprocess_img(img)
    img = img.to(device)

    # Inference: 
    y_pred = model(img)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred[0].cpu().detach().numpy()

    plt.figure(figsize=(IMG_DIM/dpi, IMG_DIM/dpi), dpi=dpi)
    plt.figimage(y_pred)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=dpi)
    buf.seek(0)

    y_pred = Image.open(buf).resize((IMG_DIM, IMG_DIM), Image.LANCZOS).convert("RGB")
    y_pred = cv2.cvtColor(np.array(y_pred), cv2.COLOR_RGB2BGR)

    cv2.imshow("Output", y_pred)

    # Record
    if record:
        cv2.imwrite(inf_out_img_src, y_pred)
        logger.info("(3) Inference Finished. Output image: {0}".format(inf_out_img_src))
    
    return y_pred

def inference_video(model, logger, record=True, dpi=500):

    # Load video and initiate video capturing
    video_source = cv2.VideoCapture(inf_vid_src)
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_source.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(inf_out_vid_src, fourcc, fps, (frame_width, frame_height))
    logger.info("(3) Video Loaded. Inferencing ... ")

    for i in range(frame_num):
        ret, frame = video_source.read()
        frame_show = frame.copy()

        frame_show = inference_image(model, logger, img=frame_show, record=False, dpi=dpi)
        
        if record:
            video_out.write(frame_show)
            if i % 20 == 19:
                logger.info("Frame {0}/{1} inferenced.".format(i+1, frame_num+1))
        
        if cv2.waitKey(1) == 27:
            break

    video_source.release()
    video_out.release()
    if record:
        logger.info("(4) Inference Finished. Output video: {0}".format(inf_out_vid_src))


if __name__ == "__main__":
    # init model
    init_msg = "(1) Initiating Inference ... "
    logger, model = select_model(model_name=net, init_msg=init_msg)

    # load model weights
    load_weights(model, best_ckpt_src, logger)

    # inference 
    inference_image(model, logger)
    # inference_video(model, logger)