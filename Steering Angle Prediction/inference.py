import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2

from models import TruckNN, TruckResnet50, TruckRNN
from config import device, best_ckpt_src, inf_img_src, inf_vid_src, inf_out_src, inf_out_img_src, inf_out_vid_src, net, seq_len
from utils import select_model, load_weights, preprocess_img
from visualize import vis_angle_on_img


def inference_image(model, logger, img=np.array(Image.open(inf_img_src)), record=True, log=True):
    
    orig_img = img.copy()
    img = preprocess_img(img, net)

    # Inference: 
    y_pred = model(img)
    angle = round(y_pred.squeeze().item(), 3)

    if log:
        logger.info("(3) Angle: {0} rad".format(angle))

    # draw angle on image 
    img = vis_angle_on_img(orig_img, angle)

    # Record
    if record:
        f = open(inf_out_src, "a")
        f.write("{0}".format(angle))
        f.close()
        cv2.imwrite(inf_out_img_src, img)
        logger.info("(3) Inference Finished. Output image: {0}".format(inf_out_img_src))
    
    return img, angle

def inference_video(model, logger, record=True, log=True):

    # Load video and initiate video capturing
    video_source = cv2.VideoCapture(inf_vid_src)
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_source.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(inf_out_vid_src, fourcc, fps, (frame_width, frame_height))
    f = open(inf_out_src, "a")
    logger.info("(3) Video Loaded. Inferencing ... ")

    for _ in range(frame_num):
        ret, frame = video_source.read()
        frame_show = frame.copy()

        frame_show, angle = inference_image(model, logger, img=frame_show, record=False, log=log)

        if record:
            f.write("{0}".format(angle))
            video_out.write(frame_show)
        
        if cv2.waitKey(1) == 27:
            break

    f.close()
    video_source.release()
    video_out.release()
    if record:
        logger.info("(4) Inference Finished. Output video: {0}".format(inf_out_vid_src))


def inference_video_seq(model, logger, record=True, log=True):

    # Load video and initiate video capturing
    video_source = cv2.VideoCapture(inf_vid_src)
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_source.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(inf_out_vid_src, fourcc, fps, (frame_width, frame_height))
    f = open(inf_out_src, "a")
    logger.info("(3) Video Loaded. Inferencing ... ")

    if frame_num < seq_len:
        raise Exception("Number of frames cannot be less than the sequence length for the RNN.")

    seq = []
    for _ in range(frame_num):
        ret, frame = video_source.read()
        frame_show = frame.copy()

        img = preprocess_img(frame, net)
        img = img.squeeze(0)
        seq.append(img)

        if len(seq) == seq_len:
            seq_tensor = np.stack(seq)
            seq_tensor = seq_tensor[np.newaxis, :]
            seq.pop(0)
            
            seq_tensor = torch.from_numpy(seq_tensor).permute([0, 2, 1, 3, 4])
            y_pred = model(seq_tensor)                                          # 1 x 5
            angle = round(y_pred.squeeze()[4].item(), 3)

            if log:
                logger.info("(3) Angle: {0} rad".format(angle))

                frame_show = vis_angle_on_img(frame_show, angle)
            
            f.write("{0}".format(angle))
            video_out.write(frame_show)
        
            if cv2.waitKey(1) == 27:
                break

    f.close()
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
    if net == "TruckRNN":
        inference_video_seq(model, logger)
    else:
    # inference_image(model, logger)
        inference_video(model, logger)
    
    