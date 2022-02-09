#import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
from sympy import true
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

#print('#############################')
#state = client.getMultirotorState()
#print(state.kinematics_estimated.position)
#print('#############################')

airsim.wait_key('Press any key to takeoff and move')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

#airsim.wait_key('Press any key to move vehicle at 5 mts/s')
client.moveToPositionAsync(212, -325, -20, 5).join() #en unity (z, x, -y, vel)
client.hoverAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

finished = False
print("LANDING START")

while not finished:
    #landing_altitude = .6
    #state = client.getMultirotorState()
    #z_p = -state.kinematics_estimated.position.z_val

    # Go to center
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    # get numpy array
    #img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8,)

    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_width = response.width
    img_height = response.height

    # original image is fliped vertically
    #img_rgb = np.flipud(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    image = img_bgr.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 15)

    #thresh = cv2.threshold(blur,0,255, cv2.THRESH_BINARY_INV)[1]
    ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("Contornos: ", len(cnts))

    centro = [0]*2
    markerInRange = False
    if len(cnts) >= 1 and len(cnts) < 10:
        for c in cnts:
            area = cv2.contourArea(c)
            #if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            #cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 3)
            centro[0] = centro[0] + ((2*x+w)/2)
            centro[1] = centro[1] + ((2*y+h)/2)

        centerAv = centro[0]/len(cnts), centro[1]/len(cnts)
        #print("Centro: ", centerAv)
        centerX = int(centerAv[0])
        centerY = int(centerAv[1])
        cv2.circle(image, (centerX, centerY), 5, (255, 0, 0), -1)
        markerInRange = True
        #cv2.drawContours(image, cnts, 0, (255,0,0), 2)
        #cv2.drawContours(image, cnts, 1, (255,0,0), 2)
        #center_publisher = Float32MultiArray(data = centerAv)
    else:
        #center_publisher = Float32MultiArray(data = [0.0, 0.0])
        print(z_p)
        print(landing_altitude)
        if z_p > landing_altitude:
            print("Not found")
        elif z_p <= landing_altitude:
            client.landAsync().join()
            client.armDisarm(False)
            finished = True
            print("***D O N E***")
            break

        markerInRange = False
    
    cv2.imshow('image', image)
    cv2.waitKey(1)
    
    #pub.publish(center_publisher)
    CENTER = centerAv
    tr = 20
    width_upper_limit = img_width/2 + tr
    width_lower_limit = img_width/2 - tr
    height_upper_limit = img_height/2 + tr
    height_lower_limit = img_height/2 - tr
    landing_altitude = .8
    state = client.getMultirotorState()
    z_p = -(state.kinematics_estimated.position.z_val+.95) #.95 fixes offset

    vx = 0
    vy = 0
    vz = 0

    if markerInRange:
        if CENTER[0] > width_lower_limit and CENTER[0] < width_upper_limit:
            if CENTER[1] > height_lower_limit and CENTER[1] < height_upper_limit:
                #disminuye altura
                if z_p > landing_altitude:
                    #vz = bl.PDz(landing_altitude, z_p, 2, .8, .8, .02)
                    vz = (landing_altitude-z_p)/2
                #if altura menor a otro landing altitude entonces land
                elif z_p <= landing_altitude:
                    #aterrizar_pub()
                    client.landAsync().join()
                    client.armDisarm(False)
                    finished = True
            else:
                #corrige en x proporcional al error (que tan lejos se esta del centro)
                e = img_height/2 - CENTER[1]
                vx = e/(img_height)
                #vx = e
                
        else:
            # corrige en y proporcional al error (que tan lejos se esta del centro)
            e = img_width/2 - CENTER[0]
            vy = e/(img_width)
            #vy = e

        #enviar_velocidad(vx, vy, vz, vaz)
        #print(-vx,-vy,-vz)
        client.moveByVelocityAsync(-vx*5,-vy*5,-vz,1).join()
        time.sleep(1)
    
airsim.wait_key('Press any key to RESET to original state and end')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
