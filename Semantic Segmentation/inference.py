import io

import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from utils import select_model, load_weights, preprocess_img, get_clustering_model, split_img
from config import best_ckpt_src, net, inf_img_src, inf_out_img_src, IMG_DIM, device, \
                    inf_vid_src, inf_out_vid_src

def inference_image(model, logger, img=np.array(Image.open(inf_img_src).convert('RGB')), compare=True, record=True, dpi=500):

    if compare:
        assert img.shape[1] == IMG_DIM * 2
        img, mask = split_img(img, IMG_DIM)

    orig_img = img.copy()
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

    y_pred_out = Image.open(buf).resize((IMG_DIM, IMG_DIM), Image.LANCZOS).convert("RGB")
    y_pred_out = cv2.cvtColor(np.array(y_pred_out), cv2.COLOR_RGB2BGR)

    # compare 
    if compare:

        # Get GT
        cluster_model = get_clustering_model(logger)
        mask = cv2.resize(mask, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
        class_map = cluster_model.predict(mask.reshape(-1, 3)).reshape(IMG_DIM, IMG_DIM)
        
        # IoU
        intersection = np.logical_and(class_map, y_pred)
        union = np.logical_or(class_map, y_pred)
        iou_score = np.sum(intersection) / np.sum(union)

        # Visualize
        class_map_out = cv2.putText(mask, 'GT, IoU: {0}'.format(round(iou_score, 3)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA) 
        y_pred_out = cv2.putText(y_pred_out, 'Prediction', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA) 
        orig_img = cv2.putText(orig_img, 'Image', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        y_pred_out = np.concatenate((orig_img, y_pred_out, class_map_out), axis=1)

    # Record
    if record:
        cv2.imwrite(inf_out_img_src, y_pred_out)
        logger.info("(3) Inference Finished. Output image: {0}".format(inf_out_img_src))
    
    cv2.imshow("Output", y_pred_out)

    return y_pred_out

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

        frame_show = inference_image(model, logger, img=frame_show, compare=False, record=False, dpi=dpi)
        
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
    model = model.to(device)

    # inference 
    # inference_image(model, logger, compare=True)
    inference_video(model, logger)