import base64

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



if __name__ == '__main__':
    app.run()


