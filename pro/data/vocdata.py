import numpy as np 
from glob import glob
import xml.etree.ElementTree as ET 
from torchvision import transforms
from PIL import Image
import torch
import pathlib
from typing import List, Dict
from pro.util.utils import find_class

class VocDataset(torch.utils.data.Dataset):

    def __init__(self, label_dir: str, image_dir: str):
        """ create a dataset for training and validation 
            Args: 
                image_dir: images folder
                label_dir: labels folder
        """
        super().__init__()
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.classes = []
        self.dataset_dict = {}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.set_label_dict()

        
    def get_label_dict(self, ) -> Dict:
        # dataset_dict['xml_filename'] = 
        #     [width, height, xmin, ymin, xamx, ymax, label_idx]

        dataset_dict= {}

        if len(self.classes) == 0:
            self.classes = find_class(self.label_dir)

        for xml_file_path in pathlib.Path(self.label_dir).glob('*.xml'): 
            xml = ET.parse(xml_file_path).getroot()
            ret = []

            for size in xml.iter("size"):     
                width = float(size.find("width").text)
                height = float(size.find("height").text)
                    
            for obj in xml.iter("object"):        
                bndbox = [width, height]        
                name = obj.find("name").text.lower().strip() 
                bbox = obj.find("bndbox")            
                pts = ["xmin", "ymin", "xmax", "ymax"]     
                for pt in pts:        
                    cur_pixel =  np.float32(bbox.find(pt).text)           
                    bndbox.append(cur_pixel)           
                label_idx = self.classes.index(name)
                bndbox.append(label_idx)    
                ret += [bndbox]
                dataset_dict[xml_file_path.name.split('.')[0]] = ret
            
        return dataset_dict

    def set_label_dict(self,): 
        if len(self.dataset_dict) == 0: 
            self.dataset_dict = self.get_label_dict()


    def __getitem__(self, index: int):
        
        """
        The __getitem__ function loads and returns a sample from the dataset at 
        the given index idx. 
        Based on the index, it identifies the image’s location on disk, 
        converts that to a tensor using read_image, retrieves the corresponding 
        label from the csv data in self.img_labels, calls the transform functions on 
        them (if applicable), and returns the tensor image and corresponding label in a 
        tuple.

        """
        
        # find the imageid (e.g., 000005.jpg, then the image id is '000005')
        self.image_ids = list(self.dataset_dict.keys())
        image_id = self.image_ids[index]

        # load the images under the specified folder, corresponding to the specified index
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = self.transform(image)
        
        # load the annotations: 
        #  1. boxes loc: [xmin, ymin, xmax, ymax]
        #  2. labels: class name (e.g., bicycle)
        #  3. area: area of the box 
        #  4. iscrowd: how many boxes are in one pic
        records = self.dataset_dict[image_id]
        boxes = torch.tensor( [i[2:6] for i in records] , dtype=torch.float32)
        labels = torch.tensor([i[-1] for i in records], dtype=torch.int64)
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
        iscrowd = torch.zeros((len(records), ), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target["image_id"] = image_id
        
        return image, target, image_id
    
    def __len__(self):
        return len(list(self.dataset_dict.keys())) 


class VocDataloader():  

    def __init__(self, dataset: VocDataset, ) -> None:

        self.dataset = dataset 

    def set_subset(self, sample_rate: float) : 
        """ Sampling the original dataset 
        """
        drawing_indices = list(range(0, self.dataset.__len__(), sample_rate))
        dataset = torch.utils.data.Subset(self.dataset, drawing_indices)
        self.dataset = dataset
        
    def split_dataset(self, split_ratio): 
        """ split the dataset into training dataset and validation dataset
        """
        torch.manual_seed(2020)
        n_train = int(self.dataset.__len__() * split_ratio)
        n_val = self.dataset.__len__() - n_train 
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_val])
            
        return train_dataset, val_dataset
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __call__(self, 
                split_ratio, 
                batch_size_train=1, 
                batch_size_val=1, 
                sample=False): 

        if sample: 
            self.set_subset(sample)
            
        train_dataset, val_dataset = self.split_dataset(split_ratio)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size_train, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size_val, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )

        return train_dataloader, val_dataloader
