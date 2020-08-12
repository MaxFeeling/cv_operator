import json, logging, os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
log = logging.getLogger("process-imageset")

"""
    label_json classification format: 
{
    "label_template": "image_classification",
    "label": [
        {
            "ID": "16585",
            "image": "/mnist_png/training/0/16585.png",
            "confidence_score": "100",
            "component": [
                {
                    "component_type": "image_tag",
                    "component_detail": {
                        "tag_value": "0"
                    }
                }
            ]
        }
    ]
}

"""


def process_imageset(label_csv):

    labels = []
    img_paths = []

    with open(label_csv) as f:
        csv_reader = csv.reader(f)
        colNames = next(csv_reader, None)
        col_orders = {name: i for i, name in enumerate(colNames)}
        for values in csv_reader:
            relative_path = values[col_orders.get('relativePath')]
            if os.name != "nt":
                img_path = os.path.abspath(
                    os.path.join(os.path.join(os.path.dirname(label_csv), "."), relative_path.lstrip("/")))
            else:
                img_path = os.path.abspath(
                    os.path.join(os.path.join(os.path.dirname(label_csv), ".."), relative_path.lstrip("/")))
            cls = values[col_orders.get('label')]
            label = eval(cls)["class"]
            img_paths.append(img_path)
            labels.append(label)

    return img_paths, labels

class ImageData(Dataset):
    """A abstract class for pre_processing imageset.
    images for training:we convert 1 channel to 3 channels,change the size to (224,224)
    and RandomHorizontalFlip
    images for evaling:we convert 1 channel to 3 channels,change the size to (224,224)
    Attributes:
        image:A file path of image
        label:A string represent the label of image
        train_mode:A string indicates we choose "train" or "val"
    """
    def __init__(self,image, label=None, train_mode = "train"):
        """
        Initialize some variables
        Load labels & names
        define transform
        """
        self.image = np.array(image)
        if label is not None:
            self.label = np.array(label)
        else:
            self.label = None
        self.train_mode = train_mode

        self.transform = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
 #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
 #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
        }

    def __len__(self):
        """
        Get the length of the entire dataset
        """
        # print("Length of dataset is ", len(self.label))
        return len(self.image)

    def __getitem__(self, idx):
        """
        Get the image item by index
        """
        image_path = self.image[idx]
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = self.transform[self.train_mode](img)
        if self.label is not None:
            label = np.array(self.label[idx]).astype(np.int)
            sample = [img, label]
            return sample
        else:
            return img
if __name__ == "__main__":
    print(process_imageset("./mnist200/labels.csv"))