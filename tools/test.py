import torch
import torch.nn as nn
import numpy as np
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

confusion_matrix = np.zeros(
        (num_classes, num_classes, 1))
accs = []

if __name__ == '__main__':
    for batch in test_dataloader:
        images, masks, edges = batch
        with torch.no_grad():    
            logits = model(images.to(device))
        upsampled_logits = nn.functional.interpolate(
            logits[0], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        confusion_matrix[..., 0] += get_confusion_matrix(
                    masks,
                    upsampled_logits,
                    masks.size(),
                    num_classes,
                    255
                )
        
        _, preds = torch.max(upsampled_logits, dim=1)
        preds[preds==1] = 0
        valid = (masks >= 0).long().cuda()
        acc_sum = torch.sum(valid * (preds == masks.cuda()).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        accs.append(acc)
        
    val_mean_accuracy = torch.mean(torch.tensor(accs))
    pos = confusion_matrix[..., 0].sum(1)
    res = confusion_matrix[..., 0].sum(0)
    tp = np.diag(confusion_matrix[..., 0])
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    metrics = {"val_mean_iou":mean_IoU, "val_mean_accuracy":val_mean_accuracy.item()}
    print(metrics)