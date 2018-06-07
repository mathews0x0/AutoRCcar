#_author__ = 'zhengwang'

import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import time
import os
import struct
import io
from PIL import Image
class CollectTrainingData(object):
    
    def __init__(self):

        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.43.41', 8050))
        self.server_socket.listen(1)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # connect to a seral port
        #self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.send_inst = True

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')

        pygame.init()
        pygame.display.set_mode((640, 480))
        self.collect_image()

    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print ('Start collecting images...')
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        # stream video frames one by one
        try:

            frame = 1
            while self.send_inst:
                    print("in loop")
                    image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

                   # if not image_len:
                    #  break
                    image_stream = io.BytesIO()
                    image_stream.write(self.connection.read(image_len))
                    image_stream.seek(0)

                    image = np.dot(np.asarray( Image.open( image_stream ), dtype='uint8' )[...,:3], [0.299, 0.587, 0.114])
                    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
                    cv2.imshow('disp', image)
                    cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)
                  #  imf=cv.imread

                    # select lower half of the image
                    roi = image[120:240, :]
                   # cv2.imshow('roi',image)
                    # save streamed images

                    
                   # cv2.imshow('roi_image', roi)


                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1

                    # get input from human driver
                    for event in pygame.event.get():
                        print("pygame get")
                        if event.type == KEYDOWN:

                            key_input = pygame.key.get_pressed()

                            # complex orders
                            if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                                print("Forward Right")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1
                              #  self.ser.write(chr(6).encode())

                            elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                                print("Forward Left")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frame += 1
                               # self.ser.write(chr(7).encode())

                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                                print("Reverse Right")
                               # self.ser.write(chr(8).encode())
                            
                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                                print("Reverse Left")
                              #  self.ser.write(chr(9).encode())

                            # simple orders
                            elif key_input[pygame.K_UP]:
                             #   self.ser.write(chr(1).encode())
                                print("Forward")
                                saved_frame += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))


                            elif key_input[pygame.K_DOWN]:
                                print("Reverse")
                                saved_frame += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[3]))
                               # self.ser.write(chr(2).encode())
                            
                            elif key_input[pygame.K_RIGHT]:
                                print("Right")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1
                               # self.ser.write(chr(3).encode())

                            elif key_input[pygame.K_LEFT]:
                                print("Left")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                             #   self.ser.write(chr(4).encode())

                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print('exit')
                                self.send_inst = False
                               # self.ser.write(chr(0).encode())
                                break
                                    
                        elif event.type == pygame.KEYUP:
                           hag=10 #  self.ser.write(chr(0).encode())

            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:    
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Streaming duration:', time0)

            print(train.shape)
            print(train_labels.shape)
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)

        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    CollectTrainingData()
