# %%
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

# %%
data_path = "/zhome/65/e/156416/E24/IDLCV/Detection_deeplearning_in_computervision/Potholes/Potholes/annotated-images"

# %%
class PotholeDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_paths = []
        self.annotations = []
        self.load_data()
        
    def load_data(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith('.xml'):
                annotation_path = os.path.join(self.data_path, filename)
                annotation = self.parse_annotation(annotation_path)
                
                image_filename = annotation['filename']
                image_path = os.path.join(self.data_path, image_filename)
                
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.annotations.append(annotation)
                else:
                    print(f"Warning: Image {image_filename} not found for annotation {filename}")
                    
    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        annotation = {
            'filename': root.find('filename').text,
            'path': root.find('path').text,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text)
            },
            'objects': []
        }
        
        for obj in root.findall('object'):
            obj_info = {
                'name': obj.find('name').text,
                'bndbox': {
                    'xmin': int(obj.find('bndbox/xmin').text),
                    'ymin': int(obj.find('bndbox/ymin').text),
                    'xmax': int(obj.find('bndbox/xmax').text),
                    'ymax': int(obj.find('bndbox/ymax').text)
                }
            }
            annotation['objects'].append(obj_info)
            
        return annotation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        return image, annotation


# %%
# Usage example:
# dataset = PotholeDataset(data_path)
# print(f"Loaded {len(dataset)} images with annotations.")
# image, annotation = dataset[0]
# print("Image shape:", image.shape)
# print("Annotation:", annotation)


# %%
# Function to plot an image with bounding boxes
def plot_image_with_bboxes(image, annotation):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for obj in annotation['objects']:
        bbox = obj['bndbox']
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']
        width = xmax - xmin
        height = ymax - ymin

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label
        ax.text(xmin, ymin - 5, obj['name'], color='red', fontsize=12, weight='bold')

    plt.title(f"Image: {annotation['filename']}")
    plt.axis('off')
    plt.show()

# # Plot the first 5 images with their bounding boxes
# for idx in range(5):
#     image, annotation = dataset[idx]
#     print(f"Image shape: {image.shape}")
#     print(f"Annotation: {annotation}")
#     plot_image_with_bboxes(image, annotation)


