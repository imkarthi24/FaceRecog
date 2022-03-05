import cv2
from simple_facerec import SimpleFacerec
import time

#result = face_recognition.compare_faces([img_encoding], img_encoding2)




# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("detected_faces/")

# Load Camera
# cap = cv2.VideoCapture(0)

# img = cv2.imread("images/crowd.jpg")

while True:
    # ret, frame = img.read() #have to get the stream frame after frame , ret is boolean  - ret is true if we have the frame and false if we don't
    frame = cv2.imread("images/amma_base.jpeg")
    rsim = cv2.resize(frame, (1200,700))


    # Detect Faces
    # this detects faces and draws rectangles and gives the face names which are taken from the title
    # face_locations = [120 , 420 ,  280 , 270] , changes every millisecond as you move your face
    # [120 , 420 ,  280 , 270] - 2 top points (left to right) and 2 bottom points (left to right)
    # [top , left , bottom, right]
    # face_locations, face_names = sfr.detect_known_faces(frame)
    face_locations, face_names = sfr.detect_known_faces(rsim)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(rsim, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2) # 1-size of the text, y1-10 is used so that it doesn't overlap on the frame
        cv2.rectangle(rsim, (x1, y1), (x2, y2), (0, 0, 200), 4) #4 is the thickness and (0,0,200) -bgr colors
        #cv2.rectangle(rsim, (554, 454), (643, 544), (0, 0, 200), 4)  # 4 is the thickness and (0,0,200) -bgr colors
        cv2.imshow("Frame", rsim)

    key = cv2.waitKey(1) # if it is 0 , waits till we press a , now it waits for a millisecond and goes for the next frame
    if key == 27: #if we press "s" on the keyboard , it breaks the loop
        break

# cap.release()  #release the camera
cv2.destroyAllWindows()