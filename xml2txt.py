import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
path = "./Car-License-Plate/annotations"
classes = {"licence":0}
labels_path="./Car-License-Plate/labels"
if not os.path.exists(labels_path):
    os.mkdir("./Car-License-Plate/labels")
#train_path=os.path.join(labels_path,"licence")
train_path=labels_path
if not os.path.exists(train_path):
    os.mkdir(train_path)
for annotations in os.listdir(path):
    dom = parse(os.path.join(path,annotations))
    root = dom.documentElement
    filename = ".txt".join(root.getElementsByTagName("filename")[0].childNodes[0].data.split(".png"))
    image_width = root.getElementsByTagName("width")[0].childNodes[0].data
    image_height = root.getElementsByTagName("height")[0].childNodes[0].data
    with open("./Car-License-Plate/labels/"+filename,"w") as r:
        for items in root.getElementsByTagName("object") :
            name = items.getElementsByTagName("name")[0].childNodes[0].data
            xmin = items.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = items.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = items.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = items.getElementsByTagName("ymax")[0].childNodes[0].data
            x_center_norm = ((int(xmin)+int(xmax)) / 2 ) / int(image_width)
            y_center_norm = ((int(ymin)+int(ymax))/2) / int(image_height)
            width_norm = ((int(xmax)-int(xmin))/int(image_width))
            height_norm = ((int(ymax)-int(ymin))/int(image_height))
            r.write(str(classes[name])+" ")
            r.write(str(x_center_norm)+" ")
            r.write(str(y_center_norm)+" ")
            r.write(str(width_norm)+" ")
            r.write(str(height_norm)+"\n")