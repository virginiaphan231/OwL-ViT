""""
In this project, we're working with COCO dataset. We use COCO minitrain 
with 25K training images (~20% of train2017 COCO)
Images are randomly sampled from full sets while trying to reserve these quantities as much as possible:
- proportion of object instances from each class,
- overall ratios of small, medium and large objects,
- per class ratios of small, medium and large objects.
Source: N. Samet, S. Hicsonmez, E. Akbas, "HoughNet: Integrating near and long-range evidence for bottom-up object detection", ECCV 2020. arXiv 2007.02355.
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class CustomCocoDataset(Dataset):
    def __init__(self, annotation_file, data_dir, is_train=False):
        super().__init__()
        self.coco = COCO(annotation_file)
        self.data_dir = data_dir
        self.is_train = is_train
        self.image_ids = list(self.coco.imgs.keys())
        
        if self.is_train:
            self.image_transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.4),
                A.Rotate(limit=45, p=0.1),
                A.RandomResizedCrop(height=448, width=448, scale=(0.8, 1.0), p= 0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'])) 
        else:
            self.image_transform = A.Compose([
                A.Resize(height=448, width=448),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.data_dir, image_info['file_name']).replace("\\", "/").replace("/", os.path.sep)
        img = Image.open(image_path).convert("RGB")

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        bboxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        text_labels = [self.coco.cats[label_id]['name'] for label_id in labels]

        img = np.array(img)

        targets = {
            'bboxes': bboxes,
            'labels': labels,
            'text_labels': text_labels
        }

        transformed = self.image_transform(image=img, bboxes=targets['bboxes'], labels=targets['labels'])
        img = transformed['image']
        target_labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        target_bboxes = torch.tensor(targets['bboxes'], dtype=torch.float32)

        return img, {"labels": target_labels, "boxes": target_bboxes, "text_labels": targets['text_labels']}
    
# Define a custom collate function to handle variable-sized data
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def load_coco_data(data_dir, annotation_file):
    coco = COCO(annotation_file)
    return coco

# # Define the paths to the COCO dataset and annotations files

# train_data_dir = r"C:\Users\admin\Desktop\VDT_Project\COCO dataset\train2017\train2017"
# test_data_dir = r"C:\Users\admin\Desktop\VDT_Project\COCO dataset\val2017"
# ann_file = r"C:\Users\admin\Desktop\VDT_Project\COCO dataset\annotations_trainval2017\annotations\instances_train2017.json"

# Define the paths to COCO minitrain 25k 
train_data_dir = r"C:\Users\admin\Desktop\VDT_Project\coco_minitrain_25k\coco_minitrain_25k\images\train2017"
test_data_dir = r"C:\Users\admin\Desktop\VDT_Project\coco_minitrain_25k\coco_minitrain_25k\images\val2017"
ann_file = r"C:\Users\admin\Desktop\VDT_Project\coco_minitrain_25k\coco_minitrain_25k\annotations\instances_minitrain2017.json"


def load_coco_data(data_dir, annotation_file):
    coco = COCO(annotation_file)
    return coco

def get_num_classes_from_coco(coco_dataset):
    # Get category information
    category_info = coco_dataset.loadCats(coco_dataset.getCatIds())

    # Get the number of classes
    num_classes = len(category_info)
    return num_classes
