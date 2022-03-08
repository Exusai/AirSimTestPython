import cv2
import numpy as np
import airsim

#connect to the AirSim simulator
#client = airsim.VehicleClient()
client = airsim.MultirotorClient()
#client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "0"
image_type = airsim.ImageType.Scene

COLOR_MIN = (0, 40, 50)
COLOR_MAX = (25, 255, 255)
t = 150 

while True:
    rawImage = client.simGetImage(camera_name, image_type)
    
    if not rawImage: pass

    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
    #png = cv2.cvtColor(png, cv2.COLOR_RGB2BGR)
    png = cv2.GaussianBlur(png, (7, 7), 0)
    hsv_img = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    result = cv2.bitwise_and(png, png, mask=mask)

    # threshold
    thresh = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1]

    # get contours
    #result = png.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        hull  = cv2.convexHull(cntr, False)
        #print(hull)
        #pts = cv2.boxPoints(rbox).astype(np.int32)
        cv2.drawContours(result, cntr, (0, 255, 0))

    cv2.imshow('png', result)
    cv2.waitKey(1)