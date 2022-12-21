import  matplotlib.pyplot as plt
from typing import List, Dict
from  pathlib import Path
import xml.etree.ElementTree as ET 
import numpy as np 
from PIL import ImageDraw, ImageFont, Image
import os 
from datetime import datetime

# faster-rcnn-lin-2.0 folder path 
DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


def gen_current_time_str(): 
    """ return a string of current time (in the format of %y%m%d-%H%M%S, e.g., 221219-123042 )
    """
    now = datetime.now()
    now_string = now.strftime("%y%m%d-%H%M%S")

    return now_string


def find_class(xml_dir: str) -> List: 
    """ fetch the class names on the dataset
        Args: 
            xml_dir: label_dir (e.g., 'pro/data/dataset/VOCdevkit/VOC2007/Annotations')
        
        Returns:
            class_name: a list contains the strings of all the class names
    """
    class_name = []
    xml_files = Path(xml_dir).glob('*.xml')
    for xml_file in xml_files: 
        xml = ET.parse(xml_file)
        for i in xml.iter('object'): 
            obj_name = i.find('name').text 
            if obj_name not in class_name: 
                class_name.append(obj_name)

    return class_name


def get_class_dict(xml_dir: str) -> Dict:
    """ generating the dict containing the pair of number of the corresponding class name
        Args: 
            xml_dir: label_dir (e.g., 'pro/data/dataset/VOCdevkit/VOC2007/Annotations')

        Returns: 
            class_name_dict: e.g., {0: people, 1: table, ...}
    """
    class_name_list = find_class(xml_dir)

    class_name_dict = {}
    for i, class_name in enumerate(class_name_list):
        class_name_dict[i] = class_name

    return class_name_dict


def plot_results(images, boxes, labels, image_id,
                class_name_dict: Dict, 
                output_dir=None, 
                scores=None, 
                gt=False): 
    
    # create the folder for storing the output
    if output_dir == None: 
        now_string = gen_current_time_str()
        # print(f'images would be output to detect/test/')
        output_dir = DIR_PATH / 'detect' / 'test' 
        if not output_dir.is_dir(): 
            os.mkdir(output_dir)

    category = class_name_dict

    if np.size(labels) == 1: 
        label = labels[0]
        box = boxes[0]
        draw = ImageDraw.Draw(images)
        label_name = category[label]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        fnt = ImageFont.truetype("arial.ttf", 10)
        if not gt: 
            label_text = "{} {:3.2f}".format(label_name,scores[0] )
        else: 
            label_text = "{}".format(label_name)

        text_w, text_h = fnt.getsize(label_text)
        draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
        draw.text((box[0], box[1]),label_text , font=fnt, fill='white')

    else: 
        for i, (label, box) in enumerate(zip(labels, boxes)):
            draw = ImageDraw.Draw(images)
            label_name = category[label]
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

            fnt = ImageFont.truetype("arial.ttf", 10)#40
            if not gt: 
                label_text = "{} {:3.2f}".format(label_name,scores[i] )
            else: 
                label_text = "{}".format(label_name)

            text_w, text_h = fnt.getsize(label_text)
            draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
            draw.text((box[0], box[1]),label_text , font=fnt, fill='white')
                   
    if not gt: 
        image_name = output_dir / "detection_id{}.png".format(image_id)
    else: 
        image_name = output_dir / "detection_id{}_gt.png".format(image_id)
    images.save(image_name)

    return output_dir
    
