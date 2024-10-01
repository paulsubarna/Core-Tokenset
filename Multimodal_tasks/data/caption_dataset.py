import os
import json
import random
from PIL import Image
from patchify import patchify, unpatchify
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from data.utils import pre_question, pre_caption
from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, img_root, train_files=[], split="train", task=None):   
        self.split = split        
         
        self.transform = transform
        self.img_root = img_root
        self.chair_root = '/app/src/BLIP/dataset/coco_chair'
        self.cup_root = '/app/src/BLIP/dataset/coco_cup'
        self.car_root = '/app/src/BLIP/dataset/coco_car'
        self.bottle_root = '/app/src/BLIP/dataset/coco_bottle'
        self.bowl_root = '/app/src/BLIP/dataset/coco_bowl'
        self.handbag_root = '/app/src/BLIP/dataset/coco_handbag'
        self.truck_root = '/app/src/BLIP/dataset/coco_truck'
        self.backpack_root ='/app/src/BLIP/dataset/coco_backpack'
        self.book_root = '/app/src/BLIP/dataset/coco_book'
        self.chair_token_root = '/app/src/BLIP/dataset/coco_token_chair'
        self.cup_token_root = '/app/src/BLIP/dataset/coco_token_cup'
        self.car_token_root = '/app/src/BLIP/dataset/coco_token_car'
        self.bottle_token_root = '/app/src/BLIP/dataset/coco_token_bottle'
        self.bowl_token_root = '/app/src/BLIP/dataset/coco_token_bowl'
        self.handbag_token_root = '/app/src/BLIP/dataset/coco_token_handbag'
        self.truck_token_root = '/app/src/BLIP/dataset/coco_token_truck'
        self.backpack_token_root ='/app/src/BLIP/dataset/coco_token_backpack'
        self.book_token_root = '/app/src/BLIP/dataset/coco_token_book'

        #[ 'coco_backpack_train', 'coco_book_train', 'coco_truck_train', 'coco_bowl_train', 'coco_bottle_train', 'coco_cup_train', 'coco_chair_train', 'coco_car_train'] #
        #['/app/src/BLIP/dataset/coco_backpack','/app/src/BLIP/dataset/coco_book', '/app/src/BLIP/dataset/coco_truck' , '/app/src/BLIP/dataset/coco_bowl',  '/app/src/BLIP/dataset/coco_bottle', '/app/src/BLIP/dataset/coco_cup', '/app/src/BLIP/dataset/coco_chair',  '/app/src/BLIP/dataset/coco_car'  ]
        #self.transform = transforms.Compose([
        #    transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        #    ])
        self.tf = transforms.Compose([ transforms.ToTensor()])
        if split=='train':
            urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
            self.annotation = []
            for f in train_files:
                #download_url(urls[f],ann_root)
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
            #self.annotation = self.annotation[ : 100000*(task+1)]
            #for i in range(task+1):
            #    self.annotation += json.load(open(os.path.join(ann_root,f'vqa_train-t{i+1}.json'),'r'))
            #if task == 0:
            #    for i in range(len(self.annotation)):
            #        self.annotation[i]['memory'] = 'NO'
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,'vqa_val.json'),'r'))    
            #json.load(open(os.path.join(ann_root,'vqa_val.json'),'r'))
            self.annotation = self.annotation[:100]
            #download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        #[ 'car', 'chair', 'dinning table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'backpack' , 'book']
        if ann['dataset']=='chair':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.chair_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.tf(image)
            else:
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
                 
 
                
            
        elif ann['dataset']=='car':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.car_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB') 
                image = self.tf(image)

            else:
                
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
                
        elif ann['dataset']=='cup':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.cup_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')
                image = self.tf(image)

            else:
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
            
        elif ann['dataset']=='bottle':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.bottle_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB') 
                image = self.tf(image)

            else:
                
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image) 
            
        elif ann['dataset']=='bowl':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.bowl_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')  
                image = self.tf(image)
                
            else:
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
                
        elif ann['dataset']=='handbag':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.handbag_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB') 
                image = self.tf(image)
                
            else:
                
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
            
        elif ann['dataset']=='truck':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.truck_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')  
                image = self.tf(image)
                
            else:
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image) 
            
        elif ann['dataset']=='backpack':
            if ann['memory'] == 'yes':
                image_path = os.path.join(self.backpack_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB') 
                image = self.tf(image)

                
            else:
                
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)
                #print(image.shape)
        elif ann['dataset']=='book':

            if ann['memory'] == 'yes':
                image_path = os.path.join(self.book_token_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')
                image = self.tf(image)
                
            else:
                
                image_path = os.path.join(self.img_root,ann['image_id'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)


        if self.split == 'train':
            caption = pre_caption(ann['caption'])   
              
                
            return image, caption




def vqa_collate_fn(batch):
    image_list, caption_list = [], []
    for image, caption in batch:
        image_list.append(image)
        caption_list.append(caption)

    return torch.stack(image_list,dim=0), caption_list      

def vqa_collate_fn_val(batch):
    image_list, question_list, question_id_list, answer_list, weight_list, n = [], [], [], [], [], []
    for image, question, question_id, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        question_id_list.append(question_id)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, question_id_list, answer_list, torch.Tensor(weight_list), n   
