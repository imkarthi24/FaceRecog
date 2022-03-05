# import the necessary packages
import base64
import np
from face_recog import RecognizeFace
from helper import convert_and_trim_bb
import imutils

import dlib
import cv2


def detectFace(encoded_data):
    detector = dlib.get_frontal_face_detector()

    npImg = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(npImg, 1)

    image = imutils.resize(img, width=900)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rects = detector(rgb, 1)

    boxes = [convert_and_trim_bb(image, r) for r in rects]
    x, y, w, h = boxes[0]

    is_success, buffer = cv2.imencode(".jpg", image[y - 50:y + h + 20, x - 25:x + w + 25])
    if is_success:
        return RecognizeFace(buffer)


