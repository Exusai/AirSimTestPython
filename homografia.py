import cv2
import numpy as np
import airsim
from sympy import false

#connect to the AirSim simulator
#client = airsim.VehicleClient()
client = airsim.MultirotorClient()
client.confirmConnection()

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
    #png = cv2.GaussianBlur(png, (7, 7), 0)
    hsv_img = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    result = cv2.bitwise_and(png, png, mask=mask)

    # threshold
    thresh = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1]

    # get contours
    img1 = png.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        """
        Box fit
        """
        #rbox  = cv2.minAreaRect(cntr)
        #pts = cv2.boxPoints(rbox).astype(np.int32)
        #cv2.drawContours(result, [pts], -1,(0, 255, 0),2)

        """
        convex hull fit
        """
        hull  = cv2.convexHull(cntr)
        cv2.drawContours(result, [hull], -1, (255, 0, 0),2)
        #print([hull])

    c = max([hull], key=cv2.contourArea)
    #c = max(contours, key=cv2.contourArea)

    # Obtain outer coordinates
    topLeft = tuple(c[c[:, :, 0].argmin()][0])
    bottomRight = tuple(c[c[:, :, 0].argmax()][0])
    topRight = tuple(c[c[:, :, 1].argmin()][0])
    bottomLeft = tuple(c[c[:, :, 1].argmax()][0])
    
    #print(len(hull))
    #print("---------------")
    # dots
    cv2.circle(result, topLeft, 4, (0, 50, 255), -1)
    cv2.circle(result, bottomRight, 4, (0, 255, 255), -1)
    cv2.circle(result, topRight, 4, (255, 50, 0), -1)
    cv2.circle(result, bottomLeft, 4, (255, 255, 0), -1)

    """
    Homografía 
    """
    P1 = np.array(([np.asarray(topLeft), np.asarray(topRight), np.asarray(bottomRight), np.asarray(bottomLeft)]))
    T = [0.0, 0.0]
    #print(P1)
    # dims reales en CM
    p2_1 = np.add([0.0, 0.0],T)
    p2_2 = np.add([122.412, 0.0],T)
    p2_3 = np.add([122.412, 200.0],T)
    p2_4 = np.add([0.0, 200.0],T)
    P2 = np.array([p2_1, p2_2, p2_3, p2_4])

    # se calcula la matriz de homografía H
    H, mask = cv2.findHomography(P1, P2, cv2.RANSAC, 5.0)
    #print(H)
    imH = cv2.warpPerspective(img1, H, (122, 200), borderMode = cv2.BORDER_CONSTANT, borderValue = (0,0,0))
    cv2.namedWindow('HomoGraph', cv2.WINDOW_NORMAL)
    cv2.imshow('HomoGraph', imH)
    #cv2.moveWindow('HomoGraph',0,0)

    org = [1,1,1] 
    vX = [1, 0, 0] 
    vY = [0, 1, 0] 
    vZ = [0, 0, 1]
    vx = vX@H
    vY = vY@H
    vZ = vZ@H
    org = org@H
    print(org/100)
    """ #client.moveByVelocityAsync(org[0], org[1], -org[2], .5)
    pt = [1,1,1]@np.linalg.inv(H)
    pt = (pt*100)
    print(int(pt[0]), int(pt[1]))
    cv2.circle(result, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)"""
    cv2.imshow('png', result) 
    cv2.waitKey(1)