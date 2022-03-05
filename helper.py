"""contains a Python function, convert_and_trim_bb,which helps to

Convert dlib bounding boxes to OpenCV bounding boxes
Trim any bounding box coordinates that fall outside the bounds of the input image"""

"""In OpenCV, bounding boxes are represented in terms of a 4-tuple of starting x-coordinate, starting y-coordinate, width, and height
Dlib represents bounding boxes via rectangle object with left, top, right, and bottom properties"""

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)