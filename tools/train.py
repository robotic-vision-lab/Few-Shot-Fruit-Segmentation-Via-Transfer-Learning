import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import utils.criterion as criterion
from models.tri_branch import TriBranch
from dataset.fruit import SemanticSegmentationDataset
from utils.criterion import BoundaryLoss
from utils.utils import get_confusion_matrix
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

batch_size = 4
num_workers = 4
num_classes = 3
lr = 7.5E-3
epochs = 100
train_path = "Minneapple"
val_path = "test_data/segmentation"

class Engine(pl.LightningModule):
    def __init__(self, epochs, lr, num_classes, train_dataloader=None, val_dataloader=None, test_dataloader=None, weights=None):
        super(Engine, self).__init__()
        self.epochs = epochs
        self.lr = lr
        
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        self.confusion_matrix = np.zeros(
        (num_classes, num_classes, 1)) 
        self.accs = []
        
        self.bd_loss = BoundaryLoss()
        
        self.fl = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor([0.005, 0.0, 0.995]),
            gamma=2,
            ignore_index=255,
            reduction='mean',
            force_reload=False
        )
        self.num_classes = num_classes
        
        self.model = TriBranch(num_classes)
        
        if weights is not None:
            self.model.load_state_dict(torch.load(weights),strict=False)

        self.bestiou = 0
        if torch.cuda.is_available():
            self.dvce = "cuda" 
        else:
            self.dvce = "cpu"
        
    def forward(self, images):
        outputs = self.model(images)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        
        images, masks, edges = batch
        
        logits = self(images)
        
        upsampled_logits = nn.functional.interpolate(
            logits[0], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        ul2 = nn.functional.interpolate(
            logits[1], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        ul3 = nn.functional.interpolate(
            logits[2], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        ubd = nn.functional.interpolate(
            logits[3], 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        l1 = self.fl(upsampled_logits, masks)
        l2 = criterion.dice_loss(masks.unsqueeze(1),upsampled_logits)
        l3 = criterion.jaccard_loss(masks.unsqueeze(1),ul3)
        
        bdl = self.bd_loss(ubd, edges)
        
        loss = l1 + l2 + .5*l3 + bdl
       
        return({'loss': loss})
    
    def validation_step(self, batch, batch_nb):
        
        images, masks, edges = batch
        outputs = self(images)
        
        logits = outputs[0]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        self.confusion_matrix[..., 0] += get_confusion_matrix(
                    masks,
                    upsampled_logits,
                    masks.size(),
                    3,
                    255
                )
        _, preds = torch.max(upsampled_logits, dim=1)
        valid = (masks >= 0).long()
        acc_sum = torch.sum(valid * (preds == masks).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        self.accs.append(acc)
        
        return({'val_loss': ""})
    
    def validation_epoch_end(self, outputs):
        val_mean_accuracy = torch.mean(torch.tensor(self.accs))
        pos = self.confusion_matrix[..., 0].sum(1)
        res = self.confusion_matrix[..., 0].sum(0)
        tp = np.diag(self.confusion_matrix[..., 0])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        if mean_IoU > self.bestiou:
            self.bestiou = mean_IoU
            torch.save(self.model.state_dict(), "best_model.pth")
            
        metrics = {"val_mean_iou":mean_IoU, "val_mean_accuracy":val_mean_accuracy.item()}
        self.accs = []
        self.confusion_matrix = np.zeros(
        (self.num_classes, self.num_classes, 1)) 
        
        print("Validation Step:")
        print(metrics)
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=self.lr, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.epochs, power=0.9)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_joint_transform = joint_transforms.Compose([
    #joint_transforms.RandomScale(),
    joint_transforms.RandomCrop((512,512)),
    joint_transforms.RandomHorizontallyFlip()
])
val_joint_transform = joint_transforms.Compose([
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

train_dataset = SemanticSegmentationDataset(train_path, joint_transform=train_joint_transform,
                                  transform=input_transform, target_transform=target_transform)
val_dataset = SemanticSegmentationDataset(val_path, joint_transform=val_joint_transform,
                                  transform=input_transform, target_transform=target_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers)

engine = Engine(
    epochs=epochs,
    lr = lr,
    num_classes=num_classes,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
)

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=engine.epochs,
    val_check_interval=len(train_dataloader),
    accumulate_grad_batches=1
)

if __name__ == '__main__':
    trainer.fit(engine)

