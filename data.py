import json
import os
from collections import Counter
import random
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
    'a photo of many {}.',
    'a photo of one {}.',
    'a photo of a {}.',
    'a photo of the {}.',
    'a good photo of a {}.',
    'a photo of the small {}.',
    'a photo of the large {}.',
    'a toy {}.',
    'a plushie {}',
    'a tattoo of the {}.',
    'a photo of a small {}.',
    'a photo of a cool {}.',
    'itap of a {}.',
    'a {} in a video game.',
    'a origami {}.',
    'a plastic {}.',
]

def get_images_dir():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        return data["images_path"]

def _add_prompt(text_label, prompt_template):
    prompted = prompt_template.replace('{}', text_label)
    return prompted

def _sample_random_prompt_templates(num_samples):
    prompt_templates = CLIP_PROMPT_TEMPLATES
    return random.choice(prompt_templates)

class AddRandomNegativeLabels:
    def __init__(self, total_num_negatives=50, labelmap_file=LABELMAP_FILE):
        self.total_num_negatives = total_num_negatives
        self.labelmap_file = labelmap_file
        self.labelmap = self._load_labelmap()  # Load the labelmap when the class is initialized
        self.sampled_negatives_dict = {}  # Dictionary to store sampled negatives

    def _load_labelmap(self):
        # Load the labelmap from the specified file
        with open(self.labelmap_file) as f:
            labelmap = json.load(f)
        return labelmap

    def _generate_sampled_negatives(self, image_id):
        if self.labelmap is None:
            raise ValueError("labelmap must be provided to create the sampled negatives.")

        # Check if sampled negatives for this image have already been generated
        if image_id not in self.sampled_negatives_dict:
            # If not, generate and store them
            all_classes = list(self.labelmap.values())

            # Filter out the classes that are already in text_labels (if defined)
            if hasattr(self, 'text_labels'):
                all_classes = [cls for cls in all_classes if cls not in self.text_labels]

            # Shuffle the list of classes and select a subset as the sampled negatives
            #random.shuffle(all_classes)
            #self.sampled_negatives_dict[image_id] = random.sample(all_classes, self.total_num_negatives)
            self.sampled_negatives_dict[image_id] = all_classes[:self.total_num_negatives]

        # Use the pre-generated sampled negatives for this image
        sampled_negatives = self.sampled_negatives_dict[image_id]

        return sampled_negatives

    def apply(self, text_labels, image_id):
        # Store the text labels for later use
        self.text_labels = text_labels

        # Use the pre-generated sampled negatives for this image
        sampled_negatives = self._generate_sampled_negatives(image_id)

        # Combine the true labels and the sampled negatives
        updated_text_labels = text_labels + sampled_negatives

        return updated_text_labels

class AddRandomPrompts:
    def __init__(self, prompt_templates):
        self.prompt_templates = prompt_templates

    def apply(self, updated_text_labels, max_queries):
        all_labels_set = list(set(updated_text_labels))
        all_labels_set.sort()

        # Modify the prompt templates to include {} for text_labels
        # modified_templates = [template.format('{}') for template in self.prompt_templates]
        fixed_template = self.prompt_templates[0]
        format_fixed_template = fixed_template.format('{}')
        fixed_templates = [format_fixed_template for _ in range(len(all_labels_set))]

        #random_templates = [random.choice(modified_templates) for _ in range(len(all_labels_set))]
        # prompted_set = [_add_prompt(label, template) for label, template in zip(all_labels_set, random_templates)]
        prompted_set = [_add_prompt(label, template) for label, template  in zip(all_labels_set, fixed_template)]

        is_promptable = [not label.startswith(NOT_PROMPTABLE_MARKER) for label in all_labels_set]
        prompted_set = [prompt if is_promptable else label for prompt, label, is_promptable in zip(prompted_set, all_labels_set, is_promptable)]

        prompted_set = [label if label != PADDING_QUERY else '' for label in prompted_set]

        #updated_prompted_text_labels = [prompted_set[all_labels_set.index(label)] for label in updated_text_labels]
        # text_queries = [template.format(label) for template, label in zip(random_templates, updated_text_labels)]
        text_queries = [template.format(label) for template, label in zip(fixed_templates, updated_text_labels)]

        # Append empty strings if len(text_queries) is less than max_queries
        while len(text_queries) < max_queries:
            text_queries.append("")
        #return updated_prompted_text_labels, text_queries
        return None, text_queries


class SingleToMultiLabel:
    def __init__(self, max_num_labels=NUM_CLASSES):
        self.max_num_labels = max_num_labels

    def apply(self, integer_labels):
        # Convert integer labels to multi-label format
        multi_label = torch.zeros((len(integer_labels), self.max_num_labels))
        for i, label in enumerate(integer_labels):
            if label < self.max_num_labels:
                multi_label[i, label] = 1  # Set the corresponding index to 1
        return multi_label

class MultiLabelToMultiHot:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes

    def apply(self, labels):
        """
        Converts multi-label labels to multi-hot representation.

        Args:
            labels: Tensor of shape [..., max_num_labels_per_instance] containing multi-labels.

        Returns:
            Tensor of shape [..., num_classes + 1] in multi-hot format.
        """
        labels = labels.clone().detach().long()
        # Labels are zero-indexed, with -1 for padding. Add 1 such that multi-hot index 0 means padding:
        labels = torch.nn.functional.one_hot(labels + 1, self.num_classes)
        labels = torch.max(labels, dim=-2)[0]  # Combine up multi-labels.

        # Update padding label such that it is 1 if there are no real labels.
        # This is necessary because multi-labels are padded, which means that the padding
        # label is initially hot for almost all real instances:
        is_padding = torch.all(labels[..., 1:] == 0, dim=-1, keepdim=True).to(labels.dtype)
        return torch.cat((is_padding, labels[..., 1:]), dim=-1)


class OwlDataset(Dataset):
    def __init__(self, annotations_file, labelmap):
        self.images_dir = get_images_dir()
        self.labelmap = labelmap
        self.label_text2id = {v:k for k, v in self.labelmap.items()}

        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]
        #print(f"Dropping {n_total - len(self.data)} examples due to no annotations")

        #self.max_promt_length = 10
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
        # print("--- Text labels: ", text_labels)
        # print("original labels shape", labels.shape)
        # print("original boxes shape", boxes.shape)
        w, h = image.size
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }
        # Apply transformations to convert the image to a tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        text_promts = self.unique(text_labels)
        text_promts_id = [self.label_text2id[t] for t in text_promts]
        new_labels = torch.tensor([text_promts_id.index(str(int(i))) for i in list(labels)])
        # print("--- Labels: ", labels, "New label:", new_labels)

        remaining = list(set(list(self.label_text2id.keys())) - set(text_promts))
        remaining = random.choices(remaining, k=self.max_promt_length - len(text_promts))
        new_text_promts = text_promts + remaining
        # print("---text prompt:", text_promts, "---new_text_prompt:", new_text_promts)

        # Use AddRandomNegativeLabels class
        # add_random_negatives = AddRandomNegativeLabels()
        # updated_text_labels = add_random_negatives.apply(text_labels, image_id = idx)
        # updated_text_labels = text_labels
        # print("labels", labels)
        # print("text_labels", len(text_labels))
        # print("updated_text_labels", len(updated_text_labels))

        # Use AddRandomPrompts class
        # add_random_prompts = AddRandomPrompts(CLIP_PROMPT_TEMPLATES)
        # _, text_queries = add_random_prompts.apply(updated_text_labels, MAX_QUERIES)
        #print("updated_prompted_text_label", len(updated_prompted_text_labels))

        # Convert labels to multi-label format
        # single_to_multi_label = SingleToMultiLabel(max_num_labels=NUM_CLASSES)
        # multi_label = single_to_multi_label.apply(labels)
        #print("multi_label length", len(multi_label))

        # Convert multi-label to multi-hot representation
        # multi_label_to_multi_hot = MultiLabelToMultiHot(num_classes=NUM_CLASSES)
        # multi_hot_label = multi_label_to_multi_hot.apply(multi_label)
        #print("multi_hot_label shape", multi_hot_label.shape)

        return image_tensor, new_labels, text_labels, boxes, new_text_promts, metadata

def get_dataloaders(
    train_annotations_file=TRAIN_ANNOTATIONS_FILE,
    test_annotations_file=TEST_ANNOTATIONS_FILE,
):
    with open(LABELMAP_FILE) as f:
        labelmap = json.load(f)

    train_dataset = OwlDataset(train_annotations_file, labelmap)
    test_dataset = OwlDataset(test_annotations_file, labelmap)


    train_labelcounts = Counter()
    for i in range(len(train_dataset)):
        train_labelcounts.update(train_dataset.load_target(i)[0])

    # scales must be in order
    # Check if scales is not empty
    if len(train_labelcounts) > 0:
        scales = []
        for i in sorted(list(train_labelcounts.keys())):
            scales.append(train_labelcounts[i])

        scales = np.array(scales)
        scales = (np.round(np.log(scales.max() / scales) + 3, 1)).tolist()
    else:
        scales = []  # Handle the case when scales is empty

    train_labelcounts = {}

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle= True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle= True, num_workers=4
    )

    return train_dataloader, test_dataloader, scales, labelmap

# Example usage:
if __name__ == '__main__':
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    
