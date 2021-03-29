# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:48:36 2021

@author: Abdelrahman
"""

import torch
import os 
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, label_dir, S = 7, B = 2, C = 20,
                 transform = None):
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                    ]
                
                boxes.append([class_label, x, y, width, height])
                
                
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image  = Image.open(img_path) 
        
        
        if self.transform:
            boxes = torch.tensor(boxes)
            image, boxes = self.transform(image, boxes)
            
            
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) 
        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)
            
            i, j = int(self.S * x), int(self.S * y)
            x_cell, y_cell = self.S * x - i, self.S * y - j
            
            width_cell, height_cell = self.S * width, self.S * height
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coord = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                    )
                label_matrix[i, j, 21:25] = box_coord
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix         
        
        
        
        

