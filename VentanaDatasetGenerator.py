import time
import airsim
import cv2
import numpy as np 
import pprint
from collections import defaultdict
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import os

def write_xml(folder, filename, bbox_list):
    root = Element('annotation',  verified='yes')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = '../dataset/' +  filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list[0]
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = e_width
    SubElement(size, 'height').text = e_height
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry
        
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = e_xmin
        SubElement(bbox, 'ymin').text = e_ymin
        SubElement(bbox, 'xmax').text = e_xmax
        SubElement(bbox, 'ymax').text = e_ymax

    #indent(root)
    tree = ElementTree(root)
    
    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)

# connect to the AirSim simulator
#client = airsim.VehicleClient()
client = airsim.MultirotorClient()
client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "0"
image_type = airsim.ImageType.Scene

# set detection radius in [cm]
client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
# add desired object name to detect in wild card/regex format
client.simAddDetectionFilterMeshName(camera_name, image_type, "ventana*")

#client.enableApiControl(True)
#airsim.wait_key('Press any key to takeoff and move')
""" print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.hoverAsync().join() """

n = 1
while True:
    rawImage = client.simGetImage(camera_name, image_type)
    
    if not rawImage:
        pass #continue

    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    img_width = png.shape[0]
    img_height = png.shape[1]
    pngOrg = png.copy()
    objects = client.simGetDetections(camera_name, image_type)
    first = False
    if objects:
        #Guardar img solo una vez
        nombre = 'IMG_' + str(n) +'.png'
        #cv2.imwrite('dataset/' + nombre, png) #Descomentar
        h, w, c = png.shape
        obs = []
        #for object in reversed(objects):
        for object in objects:
            s = pprint.pformat(object)
            #print("Ventana: %s" % s)
            # guarda xml
            obs.append((nombre, str(w), str(h), 'ventana', str(int(object.box2D.min.x_val)), str(int(object.box2D.min.y_val)), str(int(object.box2D.max.x_val)), str(int(object.box2D.max.y_val))))
            cv2.rectangle(png,(int(object.box2D.min.x_val),int(object.box2D.min.y_val)),(int(object.box2D.max.x_val),int(object.box2D.max.y_val)),(255,0,0),2) #quitar estos del
            cv2.putText(png, object.name, (int(object.box2D.min.x_val),int(object.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
            ### Codigo para volar a ventana ###
            if first:
                xMin = int(object.box2D.min.x_val)
                yMin = int(object.box2D.min.y_val)
                xMax = int(object.box2D.max.x_val)
                yMax = int(object.box2D.max.y_val)

                cropped = pngOrg[yMin:yMax, xMin:xMax]
                hsv = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)
                lower_r = np.array([160,100,100])
                upper_r = np.array([179,255,255])
                mask = cv2.inRange(hsv,lower_r,upper_r)
                res = cv2.bitwise_and(cropped,cropped,mask=mask)
                #_,thresh = cv2.threshold(res,125,255,cv2.THRESH_BINARY)
                ret, thresh = cv2.threshold(res,125,255,cv2.THRESH_BINARY)
                imgCanny = cv2.Canny(thresh, 180, 180)
                dilate =  cv2.dilate(imgCanny,None,iterations=1)
                
                #cnts, _ = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cnts, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                centro = [0]*2
                markerInRange = False
                primero = True
                for cnt in cnts:
                    if primero:
                        epsilon = 0.1*cv2.arcLength(cnt,True)
                        approx = cv2.approxPolyDP(cnt,epsilon,True)
                        x,y,w,h = cv2.boundingRect(approx)
                        #cv2.rectangle(cropped, (x, y), (x + w, y + h), (60,255,255), 3)
                        centro[0] = centro[0] + ((2*x+w)/2)
                        centro[1] = centro[1] + ((2*y+h)/2)
                        
                        if len(approx) == 4:
                            #cv2.drawContours(cropped,cnt,-1,(60,255,255),4)
                            
                            cv2.rectangle(cropped, (x, y), (x + w, y + h), (60,255,255), 3)
                            centro[0] = centro[0] + ((2*x+w)/2)
                            centro[1] = centro[1] + ((2*y+h)/2)
                        primero = False

                try:
                    centerAv = centro[0]/2, centro[1]/2
                    #print("Centro: ", centerAv)
                    centerX = int(centerAv[0])
                    centerY = int(centerAv[1])
                except:
                    centerX = int((xMax-xMin)/2) + xMin
                    centerY = int((yMax-yMin)/2) + yMin

                """ markerInRange = True
                CENTER = centerAv
                tr = 20
                width_upper_limit = img_width/2 + tr
                width_lower_limit = img_width/2 - tr
                height_upper_limit = img_height/2 + tr
                height_lower_limit = img_height/2 - tr
                landing_altitude = .8
                state = client.getMultirotorState()
                z_p = (state.kinematics_estimated.position.z_val) #.95 fixes offset

                vx = 0
                vy = 0
                vz = 0

                if markerInRange:
                    if CENTER[0] > width_lower_limit and CENTER[0] < width_upper_limit:
                        if CENTER[1] > height_lower_limit and CENTER[1] < height_upper_limit:
                            #disminuye altura
                            if z_p > landing_altitude:
                                #vz = bl.PDz(landing_altitude, z_p, 2, .8, .8, .02)
                                vx = 2
                            #if altura menor a otro landing altitude entonces land
                        else:
                            #corrige en x proporcional al error (que tan lejos se esta del centro)
                            e = img_height/2 - CENTER[1]
                            vy = e/(img_height)
                            #vx = e
                            
                    else:
                        # corrige en y proporcional al error (que tan lejos se esta del centro)
                        e = img_width/2 - CENTER[0]
                        vz = e/(img_width)
                        #vy = e

                    #enviar_velocidad(vx, vy, vz, vaz)
                    #print(-vx,-vy,-vz)
                    client.moveByVelocityAsync(vx,vy,-vz,.2).join()

                """
                cv2.circle(pngOrg, (xMin+centerX, yMin+centerY), 5, (0, 0, 255), -1) 
                first = False

        #write_xml('dataset', nombre, obs) # descomentar

    #cv2.imshow("AirSim1", pngOrg)
    cv2.imshow("AirSim2", png)
    #cv2.imshow("AirSim3", cropped)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass #break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        client.simClearDetectionMeshNames(camera_name, image_type)
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        client.simAddDetectionFilterMeshName(camera_name, image_type, "ventana*")

cv2.destroyAllWindows() 