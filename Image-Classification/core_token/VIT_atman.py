# %%
import sys,os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from PIL import Image
import torch
import time
import random
import torch.nn as nn
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import shutil
import argparse
from tqdm import tqdm 
from random import randint
import urllib
import zipfile
print(os.getcwd())
import copy
import distribute
import dataset
from patchify import patchify, unpatchify
from rtpt import RTPT
from vit import vit_base_patch16_224 as Vit
from baselines.ViT.ViT_explanation_generator import LRP, Baselines
from transformers import ViTForImageClassification, AdamW
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor, 
                                   ToPILImage)


torch.set_num_threads(25)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--num-rounds', type=int, help='to set number of rounds per task', default= 20)
parser.add_argument('--num-epochs', type=int, help='to set number of epochs per round', default= 10)
parser.add_argument('--batch-size', type=int, help='to set batch size', default= 16)
parser.add_argument('--num-class', type=int, help='to set number of class per task', default= 4)
parser.add_argument('--num-task', type=int, help='to set num task', default= 1)
parser.add_argument('--num-batches', type=int, help='to set num task', default= 16)
parser.add_argument('--acc-num', type=int, help='to set accumulation steps', default= 4)
parser.add_argument('--train-value', type=int, help='to set value of the task', default= 0)
parser.add_argument('--mem-ratio', type=float, help='to set ratio of the samples to be stored', default= 0.75)
parser.add_argument('--AD', type=float, help='percentage of dropout to be applied at the SA', default= 0.)
parser.add_argument('--ID', type=float, help='percentage of dropout to be applied at the input', default= 0.)
parser.add_argument('--PD', type=float, help='percentage of dropout to be applied at the Linear Projection', default= 0.)
parser.add_argument('--pix-ratio', type=float, help='to set ratio of the pixels to be stored', default= 0.)
parser.add_argument('--QD', type=float, help='percentage of dropout to be applied at the QKV Projection', default= 0.)
parser.add_argument('--apply-AD', type=str, help='set whether to apply dropout after self attention', default= "True" )
parser.add_argument('--apply-PD', type=str, help='set whether to apply dropout after Linear Projection inside the Tr Block', default= "True" )
parser.add_argument('--random', type=bool, help='random', default= False)#
parser.add_argument('--dropout', type=bool, help='dropout', default= False)
parser.add_argument('--device', default='cuda')
parser.add_argument('--skip', type=bool, help='skip', default= False)
parser.add_argument('--drop-val', type=bool, help='include dropout for val step', default= False)
parser.add_argument('--root', type=str, help='to set root directory', default= "/app/datasets/ILSVRC2012_imagenet" )
#parser.add_argument('--ratio', default=0.5, type=float)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=False, type=bool)
args = parser.parse_args()

#if torch.cuda.is_available():
#    device= torch.device('cuda')
#else:
#    device= 'cpu'





# ### Pretrained ViT


def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)

def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))

def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.product(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset

def unflatten(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result


def pixeling(attr_heatmap, ratio=0.1, random= False):

    b= flatten(attr_heatmap)
    b= torch.from_numpy(b)
    #print(b)
    c,r = b.sort(descending= True)
    #print(c)
    if random:
        indices = np.random.choice(np.arange(len(c)), replace=False,
                           size=int(len(c) * ratio))
        c[indices] = 0.0

    else:
        lent= int(len(b)*ratio)
        c[50176- lent: ] = 0.0

    #print(c.shape)
    b = c.gather(0, r.argsort(0))
    temp= unflatten(b,attr_heatmap)
    
    return temp

def calculate_zerop(temp):
    value = 50176 - np.count_nonzero(temp)
    
    return round((value/50176) * 100,1)


class ViTModule(nn.Module):
    def __init__(self, classes):
        super(ViTModule, self).__init__()
        self.vit = vit_LRP(pretrained=True,num_classes=classes )

    def forward(self,imgx):
        output = self.vit(imgx)
        #output= self.linear(output) 
        return output
    
    
def get_accuracy(ot, ta):
    predictions = ot.argmax(dim=1, keepdim= True).view_as(ta)
    return predictions.eq(ta).float().mean().item()


def common_step(model, imgx, label, dropout):

    logits = model(imgx, dropout= dropout)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, label)
    accuracy = get_accuracy(logits, label)

    return loss, accuracy


def training_step(model, imgx, label, optimizer, dropout):
    optimizer.zero_grad()
    loss, accuracy = common_step(model, imgx, label, dropout)  
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss, accuracy

def val_step(model,val_loader, t, dropout, device):

    v_acc= []

    for i, (img, label) in enumerate(val_loader):
        img, label= img.to(device), label.to(device)
        #if t>0:
        #    args.dropout= True
        #img= patch_embed(img, dropout= args.dropout)
        logits = model(img, dropout= dropout)
        accuracy= get_accuracy(logits, label)
        v_acc.append( accuracy)
    #print('The num batches', i)
    avg_acc= np.mean(v_acc)
    #val_loss = val_loss/(i+1)

    return  avg_acc           




def initialize_model(model, modelnew, num_cls,task):
    if task > 0:
        model.load_state_dict(torch.load(f'/app/src/Transformer_Explainability/VitABh1_{task}_{args.pix_ratio}.pt'))
    print('there')
    
    for key,value in model.state_dict().items():
        #print(key)
        if key == 'head.weight' or key == 'head.bias':
            pass
        else:
            #print(key)
            modelnew.state_dict()[key].copy_(model.state_dict()[key])
    #print(model1.state_dict())
    print('there1')
    #modelnew.to(device)
    
    return modelnew
        




def compute_otsu_threshold(attribution):
    ret, _ = cv2.threshold(attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret


def atman_pertubation(embed, model, lab, num_tokens= 196, conceptual_suppression_threshold= None , device=None):
    tokens= {}
    criterion = nn.CrossEntropyLoss()
    tokens= {'suppress_token': [],
            'additional_tokens': [],
            'loss': []}
    similarities= model.get_embedding_similarity_matrix(embed)
    out= model.process_img(embed,1, additional_indices_bool=False )
    #print(out.shape, lab.shape)
    loss= criterion(out, lab)
    #print(loss)
    B= embed.shape[0]

    with torch.no_grad():
        for i in range(num_tokens):


            temp= added(tokens['additional_tokens'])

            if i not in temp and i not in tokens['suppress_token'] :
                #print(i, 'continue')
                #continue
                #print(i, 'cont')
                similarity_scores = similarities[0][i]  #similarity score of that token plus the conceptual threshold
                additional_indices_bool = similarity_scores >= conceptual_suppression_threshold  #compute the additional indices of higher similarity
                additional_indices = additional_indices_bool.nonzero().tolist()
                additional_suppression_factors = [ 
                                model.get_suppression_factor_from_cosine_similarity(
                                    suppression_factor =0.6, 
                                    cosine_similarity = embed[0][j][i]
                                ).item()
                                for j in range(len(additional_indices))
                            ]                                                                     # compute the suppression factors of the additional indices 


                output = model.process_img(embed , B, additional_indices_bool=True, additional_indices= additional_indices, additional_suppression_factors =  additional_suppression_factors, device=device)# run one forward pass and collect logts
                loss =  criterion(output, lab)  # compute the cross_entropy loss
                #print(output.shape)
                additional_indices = list(np.array(additional_indices).flatten())
                tokens['additional_tokens'].append(additional_indices)
                tokens['suppress_token'].append(i) # remove the token along with its additional tokens
                tokens['loss'].append(loss)
                
    max_idx= tokens['loss'].index(max(tokens['loss']))
    token= tokens['suppress_token'][max_idx]
    #tokens['additional_tokens'][0]
    similarity= similarities[0][:][token]
                
    return tokens, similarity

def added(array):
    resultList= []
    for m in range(len(array)):

       # using nested for loop, traversing the inner lists
       for n in range (len(array[m])):

          # Add each element to the result list
          resultList.append(array[m][n])
    return resultList




import cv2
use_thresholding = True

def compute_otsu_threshold(attribution):
    ret, _ = cv2.threshold(attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret


def calculate_zerop(temp):
    value = 50176 - np.count_nonzero(temp)
    
    return round((value/50176) * 100,1)

def get_attribution(transformer_attribution, ratio= 0.1):

    #transform_pil= T.ToPILImage
    #transformer_attribution = attribution_generator.generate_LRP(image.unsqueeze(0).cuda(), method="transformer_attribution", index=None).detach()

    
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1
        

    
    return  transformer_attribution

def create_tokens(img):

    patches = patchify(img, (16,16,3), step=16)
    patches= patches.squeeze(2)
    patches = patches.reshape(196,16,16,3)
    return patches

def sum_nclass_idx(idx_len, start, end):
    ls= np.sum(idx_len[start:end])
    return ls

def create_trainloader(i, model, idx_len_train, images, labels, device):
    global memory_buffer_img
    global memory_buffer_label
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #print(normalize)
    _train_transforms = Compose(
            [   Resize(256), 
                CenterCrop(224),
                #RandomResizedCrop(size),
                #RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    percet = []
    #if i == 3:
    #    t_m= sum_nclass_idx(idx_len_train, start=0, end= args.num_class*(1-1))
    #    t_n= sum_nclass_idx(idx_len_train, start=0, end= args.num_class*i)
    #else:
    t_m= sum_nclass_idx(idx_len_train, start=0, end= args.num_class*(i-1))
    t_n= sum_nclass_idx(idx_len_train, start=0, end= args.num_class*i)
    unique_lb= list(set( labels[t_m:t_n]))
    unique_lb.sort()
    print(unique_lb)
    print('Start pixeling images')
    
    for ul in unique_lb:
        if ul == 0:
            t_e= sum_nclass_idx(idx_len_train, start=0, end= ul)
            a= images[:idx_len_train[ul]]
            l= labels[:idx_len_train[ul]]
            
        else:
            t_s= sum_nclass_idx(idx_len_train, start=0, end= ul-1)
            t_e= sum_nclass_idx(idx_len_train, start=0, end= ul)
            a= images[t_s:t_e]
            l= labels[t_s:t_e]
            
        idx= np.arange(len(a))
        np.random.shuffle(idx)
        arr= idx[:int(args.mem_ratio*len(idx))]        
        #idx= np.arange(int(args.mem_ratio * len(a)))
        #np.random.shuffle(idx)
        for f in tqdm(arr): #tqdm(range(length)):  #idx:
            #print(f)
            image= _train_transforms(a[f])
            #img= image.clone()
            
            image = image.unsqueeze(0)
            lab = torch.tensor(l[f]).unsqueeze(0)
            image, lab = image.to(device), lab.to(device)
            embed= model.patch_embed(image)
            _, t= atman_pertubation(embed, model, lab, num_tokens= 196, conceptual_suppression_threshold= 0.60, device=device )
            tokens= t
            image= image.squeeze(0)
            img= image.clone()
            img= img.permute(1,2,0)
            img= img.detach().cpu().numpy()
            #print(tokens.shape)
            sort , r= tokens.detach().cpu().sort(descending=True) # sort it in descending order and collect the idx
            #print(r)
            patches=  create_tokens(img)                           #patchify the image, reshape the patches into
            zero= torch.ones(16,16,3) * 0.0001  
            #r= np.arange(196)
            #random.shuffle(r)    #define the zero tensor             
            skim = r[:int(0.2* len(r))]                            # take a portion of the idx
            for idx, patch in enumerate(patches):
                if idx not in skim:
                    patches[idx, :,:,:] = zero                        #scroll through the patches to delete it if it is not present in the portion of the idx

            patches= patches.reshape(14,14,1,16,16,3)                 #unpatchify back to the image 
            img= unpatchify(patches, (224,224,3))                     #unpatchify back to the image 
            img= torch.from_numpy(img)
            img= img.permute(2,0,1)
            memory_buffer_img.append(img)
            #memory_buffer_img.append(a[f])
            memory_buffer_label.append(l[f])

 
    new_img= memory_buffer_img   
    new_labels= memory_buffer_label
    
    #with open('/app/src/Transformer_Explainability/percentage_0.5.npy', 'wb') as e:
    #    np.save(e, np.array(percet))
    f= open(f'/app/src/Transformer_Explainability/loss_continualat_{args.ID}_{args.pix_ratio}.txt', "a")
    f.write('\n'+f'[the size of the memory buffer after task {i} is {len(memory_buffer_img)}]')
    f.write('\n'+f'[the size of the New buffer after task {i} is {len(new_img)}]') 
    f.close()
    #print(len(memory_buffer_img))
    data= dataset.TinyImageNet(new_img,new_labels, transform=_train_transforms, apply_transform= False)
    if args.distributed:
        num_tasks = distribute.get_world_size()
        global_rank = distribute.get_rank()            
        #samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)    
        sampler = DistributedSampler(data, shuffle=True,num_replicas=num_tasks , rank=global_rank)
    else:
        sampler = None
    dataloader = distribute.create_loader(data, samplers= sampler, batch_size= 64, num_workers= 2, is_trains= True )
    #dataloader = DataLoader(data, batch_size=args.num_batches, shuffle= True)
    print('Finished')
    return dataloader




losses=[]
val_acc_list=[]
train_acc= []


def training_batch(modele, loader_1, optimizer,t,dropout, val_loader, device, loader_2=None):
    for e in tqdm(range(args.num_epochs)):
        if args.distributed:
            loader_1.sampler.set_epoch(e)
            if loader_2 is not None:
                loader_2.sampler.set_epoch(e)


        train_acc= []
        running_loss= 0.0

        modele.train()


        iterations = len(loader_1)
        if loader_2 is not None:
            print("Enter")
            if len(loader_1) > len(loader_2):
                print("first")
                iterations = len(loader_1)

            else:
                print("second")
                iterations = len(loader_2)


            loader2= iter(loader_2)
        loader1 = iter(loader_1)
            #_loader2 = iter(loader_2)


        l=[]
        batches= 0

        data1= None
        data2= None

        optimizer.zero_grad()
        for i in tqdm(range(iterations)):
            loss1, loss2 , acc1, acc2= 0., 0. , 0., 0.

            if loader1 is not None:
                try:
                    data1 = next(loader1)
                except StopIteration:
                    data1= None
            #data2= None
            if loader_2 is not None:
                try:
                    data2 = next(loader2)
                except StopIteration:
                    #loader2 = None #iter(loader2)
                    data2 =   None # next(loader2)

            if data1 is not None:
                batches += 1
                img1, label1= data1
                img1, label1= img1.to(device, non_blocking=True), label1.to(device)
                loss1, acc1= common_step(modele, img1, label1, dropout= True)  
                loss1 = loss1
                #acc= acc1
                del img1, label1



            if data2 is not None:
                #print('small loader')
                #model.train()
                img2, label2= data2
                img2, label2= img2.to(device), label2.to(device)
                loss2, acc2= common_step(modele,img2, label2, dropout= False)
                loss2 = loss2


                del img2, label2


            loss=  loss1+ loss2 
            acc =  acc1 +  acc2


            loss.backward()
            running_loss+= loss.item()
            #if ((i + 1) % args.acc_num == 0) or (i + 1 == iterations):
            optimizer.step()
            optimizer.zero_grad()
            #optimizer.step()


            l.append(loss.detach().cpu().item())



        losses.append(l)
        train_acc.append(acc)
        del l
        running_loss = running_loss/batches

        print(f"The Train loss after {e} epoch is: {running_loss}")
        print(f"The Train accuracy after {e} epoch is {acc}")
        modele.eval()
        val_acc = val_step(modele,val_loader[t],t, dropout= args.drop_val, device=device)
        print("The validation accuracy is:", val_acc)
        val_acc_list.append(val_acc)
        del val_acc

        #if e% args.num_epochs == 10:
        #    torch.save(model.state_dict(),f'/app/src/checkpoints/VitAB_{t}_{args.pix_ratio}_{e}.pt') 
        #    with open('/app/src/checkpoints/loss_val.npy', 'wb') as f:
        #        np.save(f, losses)



    print(f"The train accuracy after {e} epochs is: {np.mean(train_acc)}")

    torch.save(modele.module.state_dict(),f'/app/src/Transformer_Explainability/Vitat_{t}_{args.pix_ratio}.pt') 


    f= open(f'/app/src/Transformer_Explainability/loss_continualat_{args.ID}_{args.pix_ratio}.txt', "a")
    f.write('\n'+f'[the train acc for VITP for task {t} is {train_acc}]'+
            f'[the val acc for VITP for task {t} is {val_acc_list}]' + '\n')
    f.close()

def train(models, device, train_loader, val_loader, idx_len_train, images, labels):
    #optimizer =torch.optim.Adam(model.parameters(), lr= 0.0001)
    
    for t in range(args.num_task):
        
        if t > 0:
            
            print('Initializing the model')
            modelnew= models[t]
            loader2= create_trainloader(t, model.module, idx_len_train, images, labels, device)
            loader1=   train_loader[t]
            model= initialize_model(model.module,modelnew, args.num_class,t-1)
            #model.load_state_dict(torch.load(f'/app/src/Transformer_Explainability/Vitp_{t - 1}_{args.train_value}_{args.ID}.pt'))
            #vit = vit_LRP(pretrained=False, in_drop_rate= 0. ,  num_classes=  args.num_class*2)
            model.to(device)
            #print(args.gpu)
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module 
            args.dropout= False
            #train_loader[t]
            optimizer =torch.optim.Adam(model.parameters(), lr= 0.0001)
            #optimizer = torch.optim.Adam(vit.parameters(), lr= 0.0001)
            training_batch(model, loader1, optimizer,t, args.dropout,val_loader, device, loader2)

        

        else:
            model= models[t]
            model_without_ddp = model
            model.to(device)
            #if t  ==0:
            #model.load_state_dict(torch.load(f'/app/src/Transformer_Explainability/VitAB_0_0.8.pt', map_location= torch.device('cuda')))
            #else:
            #    model.load_state_dict(torch.load(f'/app/src/Transformer_Explainability/VitABr_{t}_0.8.pt', map_location= torch.device('cuda')))
            print(args.gpu)
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module 

            
            
            #model.to(device)
            args.dropout= True
            loader= train_loader[t]
            
            optimizer =torch.optim.Adam(model.parameters(), lr= 0.0001)
            training_batch(model, loader, optimizer,t, args.dropout, val_loader, device)
            
                
def main():
    models= []
    
            
    distribute.init_distributed_mode(args)    
    
    device = torch.device(args.device)
    

    #"/app/datasets/ILSVRC2012_imagenet"
    # Define training and validation data paths

    # fix the seed for reproducibility
    rank= distribute.get_rank()
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    DATA_DIR= args.root
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    
    for t in range(args.num_task):
        #models.append(VIT(pretrained=True,in_drop_rate= args.ID, num_classes=  args.num_class * (t+1)))
        if t == 0:
            models.append(Vit(pretrained=True,in_drop_rate= args.ID, num_classes=  args.num_class * (t+1)))
        else:
            models.append(Vit(pretrained=False,in_drop_rate= args.ID, num_classes=  args.num_class * (t+1)))


    train_loader, val_loader, idx_len_train, images, labels = dataset.create_loader(args, DATA_DIR, TRAIN_DIR, VALID_DIR, rank)

    train(models, device, train_loader, val_loader, idx_len_train, images, labels)
    
    
    
if __name__ == '__main__':
    #device= 'cuda'

    rtpt = RTPT(name_initials='SP', experiment_name='TestingTF', max_iterations=1000)
    # Start the RTPT tracking
    rtpt.start()
    percet = []
    timestamp1 = time.time()
    f= open(f'/app/src/Transformer_Explainability/loss_continualat_{args.ID}_{args.pix_ratio}.txt', "a")
    f.write('\n'+'-------------------------------------------------------'+ '\n' +
            f'[The HYPERPARAMETERS for process is {args.num_task,  args.num_class } and ID-{args.ID}, pix-ratio-{args.pix_ratio}]' + '\n')
    f.close()
    memory_buffer_img= []
    memory_buffer_label= []
    

    #model= VIT(pretrained= True, num_classes= args.num_class)
    
    main()
    print('Finished')
    timestamp2 = time.time()
    print('Total time:', timestamp2- timestamp1)
