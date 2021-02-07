import cv2 
from math import tan, pi 

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
