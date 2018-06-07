import struct
import io
from PIL import Image

__author__ = 'zhengwang'

import threading
import SocketServer
import serial
import cv2
import numpy as np
import math

class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, 4])
        self.model.create(layer_size)
        self.model.load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

class RCControl(object):

    def __init__(self):
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

    def steer(self, prediction):
        if prediction == 2:
            self.serial_port.write(chr(1))
            print("Forward")
        elif prediction == 0:
            self.serial_port.write(chr(7))
            print("Left")
        elif prediction == 1:
            self.serial_port.write(chr(6))
            print("Right")
        else:
            self.stop()

    def stop(self):
        self.serial_port.write(chr(0))


class VideoStreamHandler(SocketServer.StreamRequestHandler):





    # create neural network
    model = NeuralNetwork()
    model.create()
    rc_car = RCControl()


    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')





    def handle(self):

        global sensor_data
        stream_bytes = ' '
        stop_flag = False
        stop_sign_active = True

        # stream video frames one by one
        try:
            while True:

                image_len = struct.unpack('<L', self.rfile.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.rfile.read(image_len))
                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)
                imag = Image.open(image_stream)
                try:
                    img = np.asarray(imag, dtype='uint8')
                except SystemError:
                    img = np.asarray(imag.getdata(), dtype='uint8')

                # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
                #   image = 0.2989 * r + 0.5870 * g + 0.1140 * b
                image = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                # tmp = np.fromstring(imag, dtype='np.uint8')

                # gray = cv2.imdecode(tmp, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)

                # print(type(img))
                # print(type(img))
                # lower half of the image
                half_gray = image[120:240, :]




                cv2.imshow('image', img)
                # cv2.imshow('mlp_image', half_gray)

                # reshape image
                image_array = half_gray.reshape(1, 38400).astype(np.float32)

                # neural network makes prediction
                prediction = self.model.predict(image_array)

                # stop conditions


                self.rc_car.steer(prediction)
                self.stop_start = cv2.getTickCount()
                self.d_stop_sign = 25

                if stop_sign_active is False:
                    self.drive_time_after_stop = (self.stop_start - self.stop_finish) / cv2.getTickFrequency()
                    if self.drive_time_after_stop > 5:
                        stop_sign_active = True

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.rc_car.stop()
                    break

            cv2.destroyAllWindows()

        finally:
            print "finally"


class ThreadServer(object):
    def server_thread(host, port):
        print("in threadserver")
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()


    video_thread = threading.Thread(target=server_thread('192.168.43.41', 8004))
    video_thread.start()


if __name__ == '__main__':
    ThreadServer()
