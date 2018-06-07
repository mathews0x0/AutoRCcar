from time import sleep
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (320, 240)
camera.start_preview()
# Camera warm-up time
sleep(2)
camera.capture('1.jpg')
