import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

root = Element("annotation", verified='yes')
folder = ET.SubElement(root, "train").text='images'
filename = ET.SubElement(root, "filename").text='raccoon-1.jpg'

print(ET.tostring(root))