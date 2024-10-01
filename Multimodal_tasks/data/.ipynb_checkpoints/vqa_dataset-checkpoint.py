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
from data.utils import pre_question
from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root,mem_root, doc_root , train_files=[], split="train", task=None, blip = None):  
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.mem_root = mem_root
        self.doc_root = doc_root
        self.blip= blip
        #self.transform = transforms.Compose([
        #    transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        #    ])
        
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
    
    def patchify(self, IMG):


        patches = patchify(IMG, (16,16,3), step=16)
        patches= patches.squeeze(2)
        patches = patches.reshape(900,16,16,3)
        return patches

    
    def create_token(self, annot, image):
            #question= annot['question']
            #ids= annot['question_id']
            #answer= annot['answer']
            image = image.unsqueeze(0)

            ran = np.arange(900)
            random.shuffle(ran)
            img= image[0].clone()
            img= img.permute(1,2,0)
            img= img.detach().cpu().numpy()


            patches=  self.patchify(img)                           #patchify the image, reshape the patches into
            zero= torch.ones(16,16,3) * 0.0001                      #define the zero tensor             
            skim = ran[:int(0.5* len(ran))]                            # take a portion of the idx
            for idx, patch in enumerate(patches):
                if idx not in skim:
                    patches[idx, :,:,:] = zero                        #scroll through the patches to delete it if it is not present in the portion of the idx

            patches= patches.reshape(30,30,1,16,16,3)                 #unpatchify back to the image 
            img= unpatchify(patches, (480,480,3))  
            img= torch.from_numpy(img)
            img= img.permute(2,0,1)
            return img
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if ann['dataset']=='vqa':
            if ann['memory'] == 'yes':
                #image_path = os.path.join(self.mem_root,ann['image_memory']) 
                #image = torch.load(os.path.join(self.mem_root,ann['image_memory']) )
                transform = transforms.Compose([ transforms.ToTensor()])
                image_path = os.path.join(self.mem_root,ann['image_memory'])  
                image = Image.open(image_path).convert('RGB')   
                image = transform(image) 
                #image = self.create_token(ann, image)
 
            else:
                image_path = os.path.join(self.vqa_root,ann['image'])  
                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image) 
                
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)    
            
        elif ann['dataset']=='docvqa':
            image_path = os.path.join(self.doc_root,ann['image'])  
            
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)  
            
        elif ann['dataset']=='CLV':
            if ann['memory'] == 'yes':
                #image_path = os.path.join(self.mem_root,ann['image_memory']) 
                #image = torch.load(os.path.join('/app/src/BLIP/dataset/token_image_CLV',ann['image_memory']) )
                transform = transforms.Compose([ transforms.ToTensor()])
                image_path = os.path.join(self.doc_root,ann['image_memory'])  

                image = Image.open(image_path).convert('RGB')   
                image = transform(image) 

 
            else:
                image_path = os.path.join('/app/src/BLIP/dataset/CLEVR_v1.0/images/train',ann['image'])  

                image = Image.open(image_path).convert('RGB')   
                image = self.transform(image)  
                image= image[:3, :,:]


        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']   
            
                        
            if ann['dataset']=='vqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg' :
                answers = [ann['answer']]
                weights = [0.2]  
                
            return image, question, question_id, answers, weights


        elif self.split=='train':                       
            
            question = pre_question(ann['question'])        
            
            if ann['dataset']=='vqa' or ann['dataset']=='docvqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.5]  
                
            elif ann['dataset']=='CLV':
                answers = [ann['answer']]
                weights = [1] 
                
            #elif ann['dataset']=='docvqa':
            #    answers = ann['answer']
            #    weights = [0.2]  

            return image, question, answers, weights


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        

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
