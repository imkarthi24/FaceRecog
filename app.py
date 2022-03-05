import base64
import os

from flask import Flask,request
from face_detection import detectFace
app = Flask(__name__)

@app.route('/health',methods = ['GET'])
def health():
    return 'Healthy'

def recognizeFace(byte_data):
    return detectFace(byte_data)

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.files['image']
    image_data = base64.b64encode(data.read())

    return recognizeFace(image_data)


def storeFaceData(image,name):
    fileName = name + ".jpg"
    image.save(os.path.join("Training_images/", fileName))
    return "Ok"


@app.route('/storeimage', methods=['POST'])
def storeImage():
    data = request.files['image']
    name = request.args.get('name')
    return storeFaceData(data, name)



if __name__ == '__main__':
    app.run()


