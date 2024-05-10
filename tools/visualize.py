import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from models.tri_branch import TriBranch
from dataset.fruit import SemanticSegmentationDataset
from utils.utils import get_confusion_matrix
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
num_classes = 3
num_workers = 4
weights_path = "path/to/weights"
dataset_path = "test_data/segmentation"
output_path = "visualizations"
color_map = {
   0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 255),
    255: (0, 0, 0)
}

def prediction_to_vis(prediction):
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape)
    for i,c in color_map.items():
        vis[prediction == i] = color_map[i]    
    return Image.fromarray(vis.astype(np.uint8))

model = TriBranch(num_classes).to(device)
model.load_state_dict(torch.load(weights_path),strict=False)
model.eval()


test_joint_transform = joint_transforms.Compose([
])
input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
target_transform = extended_transforms.MaskToTensor()
restore_transform = standard_transforms.Compose([
    extended_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])


test_dataset = SemanticSegmentationDataset(dataset_path, joint_transform=test_joint_transform,
                                  transform=input_transform, target_transform=target_transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers)

image_file_names = sorted([f for f in os.listdir(dataset_path + "/images") if '.png' in f])

if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, batch in enumerate(test_dataloader):
        images, masks, edges = batch
        with torch.no_grad():    
            logits = model(images.to(device))
        upsampled_logits = nn.functional.interpolate(
            logits[0], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
        masks = masks.cpu().numpy()
        im = prediction_to_vis(predicted_mask[0,:,:])
        im.putalpha(75)
        img = restore_transform(images[0])
        img.putalpha(255)
        Image.alpha_composite(img, im).save(os.path.join(output_path, image_file_names[i]))