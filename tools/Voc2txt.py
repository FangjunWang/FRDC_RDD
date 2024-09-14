import xml.etree.ElementTree as ET
import pickle
import os
import cv2
import numpy as np
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm
import shutil

classes = ['D00', 'D10', 'D20', 'D40', 'Repair']
def convert(size, box, mode="xyxy"):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x1 = box[0]
    x2 = box[1]
    y1 = box[2]
    y2 = box[3]
    if mode == "xyxy":
        return (x1, y1, x2, y2)
    cx = (box[0] + box[1])/2.0 - 1
    cy = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    if mode == "xywh":
        return (x1, y1, w, h)
    if mode == "cxcywh":
        return (cx, cy, w, h)
    if mode == "dxdydwdh":
        x = x1*dw
        w = w*dw
        y = y1*dh
        h = h*dh
        return (x,y,w,h)
    return (cx*dw, cy*dh, w*dw, h*dh)

def convert_annotation(xml_name, txt_name, jpg_name=None, mode="xyxy"):
    in_file = open(xml_name, 'r', encoding='utf-8')
    out_file = open(txt_name, 'w', encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    #print(root)
    if jpg_name:
        img = cv2.imdecode(np.fromfile(jpg_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        #img = cv2.imread(jpg_name)
        h, w = img.shape[:2]
    else:
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    #print(h, w)
    for obj in root:#.iter('object'):
        if obj.tag != 'object':
            continue
        #print(obj)
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        #print(cls)
        #if cls == "motorbike":
        #    cls = "person"
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b, mode=mode)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    xml_dir = "/data2/wangfj/data/road_damage/train/xmls"
    txt_dir = "/data2/wangfj/data/road_damage/train/labels"
    for name in tqdm(os.listdir(xml_dir)):
        # for name in files:
        xml_name = os.path.join(xml_dir, name)
        if xml_name.endswith(".xml"):
            txt_name = os.path.join(txt_dir, name.replace(".xml", ".txt"))
            convert_annotation(xml_name, txt_name, jpg_name=None, mode="")
                    
