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
client = airsim.VehicleClient()
client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "0"
image_type = airsim.ImageType.Scene

# set detection radius in [cm]
client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
# add desired object name to detect in wild card/regex format
client.simAddDetectionFilterMeshName(camera_name, image_type, "ventana*")


n = 1
while True:
    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        pass #continue
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    objects = client.simGetDetections(camera_name, image_type)
    if objects:
        #Guardar img solo una vez
        nombre = 'IMG_' + str(n) +'.png'
        cv2.imwrite('dataset/' + nombre, png)
        h, w, c = png.shape
        obs = []
        for object in objects:
            s = pprint.pformat(object)
            #print("Ventana: %s" % s)
            # guarda xml
            obs.append((nombre, str(w), str(h), 'ventana', str(int(object.box2D.min.x_val)), str(int(object.box2D.min.y_val)), str(int(object.box2D.max.x_val)), str(int(object.box2D.max.y_val))))
            # Display images
            cv2.rectangle(png,(int(object.box2D.min.x_val),int(object.box2D.min.y_val)),(int(object.box2D.max.x_val),int(object.box2D.max.y_val)),(255,0,0),2)
            cv2.putText(png, object.name, (int(object.box2D.min.x_val),int(object.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))

        write_xml('dataset', nombre, obs)


    cv2.imshow("AirSim", png)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass #break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        client.simClearDetectionMeshNames(camera_name, image_type)
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        client.simAddDetectionFilterMeshName(camera_name, image_type, "ventana*")

cv2.destroyAllWindows() 