import json
import os
from collections import Counter
import random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from transformers import OwlViTProcessor
from collections import Counter

# Constants
TRAIN_ANNOTATIONS_FILE = "data/train.json"
TEST_ANNOTATIONS_FILE = "data/test.json"
LABELMAP_FILE = "data/labelmap.json"

NOT_PROMPTABLE_MARKER = '#'
PADDING_QUERY = ''

MAX_QUERIES = 20 #10#80
NUM_CLASSES = 80 # no-object label will be added during multi-hot encoding 

CLIP_PROMPT_TEMPLATES = [
    '{}.',
    'a bad photo of a {}.',
    'a photo of one {}.',
    'a photo of a {}.',
    'a photo of the {}.',
    'a good photo of a {}.',
    'a photo of the small {}.',
    'a photo of the large {}.',
    'a cropped photo of a {}.',
    'a drawing of a {}',
    'a tattoo of the {}.',
    'a photo of a small {}.',
    'a photo of a cool {}.',
    'a photo of my {}.',
    'a photo of a weird {}.',
    'a blurry photo of the {}.',
    'a close-up photo of the {}.',
]

def get_images_dir():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        return data["images_path"]

def _add_prompt(text_label, prompt_template):
    prompted = prompt_template.replace('{}', text_label)
    return prompted


class AddRandomPrompts:
    def __init__(self, prompt_templates):
        self.prompt_templates = prompt_templates

    def apply(self, updated_text_labels):

        # Modify the prompt templates to include {} for text_labels
        modified_templates = [template.format('{}') for template in self.prompt_templates]

        random_templates = [random.choice(modified_templates) for _ in range(len(updated_text_labels))]
        prompted_set = [_add_prompt(label, template) for label, template in zip(updated_text_labels, random_templates)]
        

        is_promptable = [not label.startswith(NOT_PROMPTABLE_MARKER) for label in updated_text_labels]
        prompted_set = [prompt if is_promptable else label for prompt, label, is_promptable in zip(prompted_set, updated_text_labels, is_promptable)]

        prompted_set = [label if label != PADDING_QUERY else '' for label in prompted_set]

        text_queries = [template.format(label) for template, label in zip(random_templates, updated_text_labels)]
    
        return None, text_queries


class OwlDataset(Dataset):
    def __init__(self, annotations_file, labelmap, is_train):
        self.images_dir = get_images_dir()
        self.labelmap = labelmap
        self.is_train = is_train
        self.label_text2id = {v:k for k, v in self.labelmap.items()}

        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]
        #print(f"Dropping {n_total - len(self.data)} examples due to no annotations")

        self.max_promt_length = 20

    def load_image(self, idx: int) -> Image.Image:
        url = list(self.data[idx].keys())[0]
        path = os.path.join(self.images_dir, os.path.basename(url))
        image = Image.open(path).convert("RGB")
        return image, path
    
    def load_target(self, idx: int):
        annotations = list(self.data[idx].values())[0]

        labels = []
        boxes = []
        text_labels = []  # Initialize a list to store text labels

        for annotation in annotations:
            label = annotation["label"]
            text_label = self.labelmap[str(label)]  # Get text label as a string

            if text_label is not None:
                labels.append(label)
                text_labels.append(text_label)
                boxes.append(annotation["bbox"])

        # Convert labels and boxes to tensors
        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return labels, boxes, text_labels

    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes, text_labels = self.load_target(idx)
        w, h = image.size
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }
        

        text_promts = self.unique(text_labels)
        text_promts_id = [self.label_text2id[t] for t in text_promts]
        new_labels = torch.tensor([text_promts_id.index(str(int(i))) for i in list(labels)])

        remaining = list(set(list(self.label_text2id.keys())) - set(text_promts))
        remaining = random.choices(remaining, k=self.max_promt_length - len(text_promts))
        new_text_promts = text_promts + remaining

        # Use AddRandomPrompts class
        add_random_prompts = AddRandomPrompts(CLIP_PROMPT_TEMPLATES)
        _, text_queries = add_random_prompts.apply(new_text_promts)

        # Apply augmentation transformation
        if self.is_train:
            transform = transforms.Compose([transforms.Resize(size= (768,768)),
                                            transforms.RandomHorizontalFlip(p = 0.2),
                                            transforms.RandomRotation(degrees = 30),
                                            transforms.ToImageTensor(),
                                            ])
        else:
            transform = transforms.Compose([transforms.Resize(size=(768,768)),
                                            transforms.ToImageTensor()])
        
        
        image_tensor, boxes, new_labels = transform(image, boxes, new_labels)
    

        # OwlDataset returns images, new_labels, boxes, text_queries, metadata as tensors
        return image_tensor, new_labels, boxes, text_queries, metadata

# def custom_collate(batch):
#     images, labels, boxes, text_queries, metadata = zip(*batch)
#     # Stack the image tensors into a batch
#     images = torch.stack(images)

#     # Find the maximum length of labels in this batch
#     max_len = max(len(label) for label in labels)
#     # Create a padded_labels list to hold the padded labels
#     padded_labels = []
    
#     # Pad each label sequence with zeros to match the maximum length
#     for label in labels:
#         padding_length = max_len - len(label)
#         padded_label = np.pad(label, (0, padding_length), 'constant', constant_values=0)
#         padded_labels.append(padded_label)
    
#     # Stack the padded labels into a tensor
#     labels = torch.tensor(padded_labels)

#     # Find the maximum number of bounding boxes in this batch
#     max_num_boxes = max(boxes.shape[0] for boxes in boxes)
    
#     # Create a padded_boxes list to hold the padded bounding boxes
#     padded_boxes = []
    
#     # Pad each bounding box sequence with zeros to match the maximum number of boxes
#     for box_tensor in boxes:
#         num_boxes = box_tensor.shape[0]
#         padding_rows = max_num_boxes - num_boxes
#         if padding_rows > 0:
#             padding = torch.zeros((padding_rows, 4), dtype=torch.float32)
#             padded_box_tensor = torch.cat((box_tensor, padding), dim=0)
#         else:
#             padded_box_tensor = box_tensor
#         padded_boxes.append(padded_box_tensor)
    
#     # Stack the padded boxes into a tensor
#     boxes = torch.stack(padded_boxes)

#     # # Stack text_queries list into a batch
#     # text_queries = torch.tensor(text_queries)

#     # # Stack metadata list into a batch
#     # metadata = torch.tensor(metadata)

#     return images, labels, boxes, text_queries, metadata


def get_dataloaders(
    train_annotations_file=TRAIN_ANNOTATIONS_FILE,
    test_annotations_file=TEST_ANNOTATIONS_FILE,
):
    with open(LABELMAP_FILE) as f:
        labelmap = json.load(f)

    train_dataset = OwlDataset(train_annotations_file, labelmap, is_train= True)
    test_dataset = OwlDataset(test_annotations_file, labelmap, is_train= False)


    train_labelcounts = Counter()
    for i in range(len(train_dataset)):
        train_labelcounts.update(train_dataset.load_target(i)[0])

    # scales must be in order
    scales = []
    for i in sorted(list(train_labelcounts.keys())):
        scales.append(train_labelcounts[i])

    scales = np.array(scales)
    scales = (np.round(np.log(scales.max() / scales) + 3, 1)).tolist()
    
    train_labelcounts = {}

    train_dataloader = DataLoader(
        train_dataset, batch_size= 1, shuffle= True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size= 1, shuffle= True, num_workers=4)

    return train_dataloader, test_dataloader, scales, labelmap

# Example usage:
if __name__ == '__main__':
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    
