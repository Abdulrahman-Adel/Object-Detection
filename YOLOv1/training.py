# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:27:03 2021

@author: Abdelrahman
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (intersection_over_union, mean_avg_precision, non_max_suppression
                    ,plot_image, get_bboxes, cellboxes_to_boxes)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

DEVICE = torch.device("cpu")

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
#CSV_FILE = "data/8examples.csv"
CSV_FILE = "data/100examples.csv"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image), bboxes
            
        return image, bboxes    
            
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) 

def train(dataloader, model, optimizer, loss_fn):
    loop = tqdm(dataloader, leave = True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE) 

        output = model(x)  
        loss = loss_fn(output, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item()) 
        
    print(f"Mean Loss was {sum(mean_loss)/len(mean_loss)}")
    


def main():

    model = Yolov1(split_size = 7, num_boxes = 2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 2e-5, weight_decay = 0)
    loss_fn = YoloLoss()
    
    
    train_dataset = VOCDataset(CSV_FILE,
                               transform = transform,
                               image_dir = IMG_DIR,
                               label_dir = LABEL_DIR)
    
    train_loader = DataLoader(train_dataset,
                              num_workers=0,
                              batch_size=16,
                              shuffle = True)
    
    
    EPOCHS = 100
    for epoch in range(EPOCHS):
        
        print(f"Number of Epoch: {epoch+1}")
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_avg_precision(
            pred_boxes = pred_boxes, true_boxes = target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
        print(f"Train mAP: {mean_avg_prec}")
        
        train(train_loader, model, optimizer, loss_fn)
             
    """for x, y in train_loader:
             x = x.to(DEVICE)
             for idx in range(8):
                 bboxes = cellboxes_to_boxes(model(x))
                 bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                 plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes, f"image{idx+1}.jpg")

             import sys
             sys.exit()    """
    torch.save(model.state_dict(), 'Yolov1_overfitting.pt')          

if __name__ == "__main__":
    main()
    