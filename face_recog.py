import base64

import cv2
import np
from simple_facerec import SimpleFacerec


# Encode faces from a folder of known faces
# This needs to be  high res images of the faces we need to recognize

sfr = SimpleFacerec()
sfr.load_encoding_images("Training_images/")


def RecognizeFace(image_data):
    #npImg = np.fromstring(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(image_data, 1)
    rsim = cv2.resize(img, (400,400))
    cv2.imshow("Frame-2", rsim)

    face_locations, face_names = sfr.detect_known_faces(rsim)

    return face_names[0]




