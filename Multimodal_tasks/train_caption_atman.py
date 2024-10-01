'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
#import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import logging
from pathlib import Path
from rtpt import RTPT
import torch
from patchify import patchify, unpatchify
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from coreset.cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader
from coreset.modal.modeling import VisionTransformer, CONFIGS
from coreset.cords.utils.config_utils import load_config_data
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor, 
                                   ToPILImage)
from models.blip import blip_decoder
import utils
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from data.caption_dataset import vqa_dataset, vqa_collate_fn

def create_tokens(img):

    patches = patchify(img, (16,16,3), step=16)
    patches= patches.squeeze(2)
    patches = patches.reshape(576,16,16,3)
    return patches


def select_coreset(model, file, val_file, transform, args):
    #if task ==1:
    datasets= vqa_dataset( transform, ann_root= '/app/src/BLIP/dataset/captions' ,img_root= '/app/src/BLIP/dataset/coco_train2014/train2014' , train_files=[file], split="train", task=0)
    val_datasets= vqa_dataset( transform, ann_root= '/app/src/BLIP/dataset/captions' ,img_root= '/app/src/BLIP/dataset/coco_train2014/val2014/' , train_files=[val_file], split="test", task=0)

    train_loader=utils.create_loader(datasets,samplers=None,batch_size=2,num_workers=1,is_trains=True, collate_fns=vqa_collate_fn)
    val_loader=utils.create_loader(val_datasets,samplers=None,batch_size=2,num_workers=1,is_trains=False, collate_fns=vqa_collate_fn)
        
    
        
        
    cfg= load_config_data('set the path to the coreset')

    logger = logging.getLogger(__name__)

    dataloader = GradMatchDataLoader(model, train_loader, val_loader, cfg, logger,
                                        batch_size=cfg['train_batch_size'],
                                        shuffle=cfg['shuffle'],
                                        pin_memory= True)
    idx, w= dataloader._resample_subset_indices()
    
    np_idx = np.array(idx)
    with open('/app/src/BLIP/index_cs10.npy', 'wb') as f:
        np.save(f, np_idx)

    
    return idx


def update_annotation(model, ann_root, file,val_file, root, tok_root, transform, args):
    cr_idx =  select_coreset(model, file, val_file,  transform, args)

    annotation = json.load(open(os.path.join(ann_root,f'{file}.json'),'r'))
    img_root = root

    for i in tqdm(range(len(annotation))):
        #if i > 18683:
        if i in cr_idx:
            index= annotation[i]['id']
            caption= annotation[i]['caption']
            image =Image.open(os.path.join(img_root, annotation[i]['image_id'])).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0).cuda()

            attn= model.visual_encoder.forward_atman(image)
            attn= attn.reshape(576)
            tokens= attn
            img= image[0].clone()
            img= img.permute(1,2,0)
            img= img.detach().cpu().numpy()

            sort , r= tokens.detach().cpu().sort(descending=True) # sort it in descending order and collect the idx
            patches=  create_tokens(img)                           #patchify the image, reshape the patches into
            zero= torch.ones(16,16,3) * 0.0001                      #define the zero tensor             
            skim = r[:int(args.ratio* len(r))]                            # take a portion of the idx
            for idx, patch in enumerate(patches):
                if idx not in skim:
                    patches[idx, :,:,:] = zero                        #scroll through the patches to delete it if it is not present in the portion of the idx

            patches= patches.reshape(24,24,1,16,16,3)                 #unpatchify back to the image 
            img= unpatchify(patches, (384,384,3))  
            img= torch.from_numpy(img)
            img= img.permute(2,0,1)


            save_image(img, os.path.join(tok_root, f'coco_{i}.png'))
            annotation[i]['image_memory']= f'coco_{i}.png'
            annotation[i]['memory']= 'NO'

            del image
            torch.cuda.empty_cache()

    

    temp_annot= []
    for ids in tqdm(range(len(annotation))):
        if ids in cr_idx:
            temp_annot.append(annotation[ids])

    with open(os.path.join(ann_root,f'{file}_cr10.json'), "w") as jsonFile:
        json.dump(temp_annot, jsonFile)
    annota = json.load(open(os.path.join(ann_root,f'{file}_cr10.json'),'r'))
    print(len(annota))
    with open(os.path.join(ann_root,f'{file}.json'), "w") as jsonFile:
        json.dump(annotation, jsonFile)


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50
    optimizer.zero_grad()
    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       
        
        loss = model(image, caption)    
        loss = loss/2
        loss.backward()
        if (i+1)%2 == 0:
            optimizer.step()  
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    transform = transforms.Compose([
            transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
    #### Dataset #### 
    print("Creating captioning dataset") #config['pretrained']
    

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=None, image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    model.load_state_dict(torch.load('/app/src/BLIP/output/BLIP_LPd_0_49.pt', map_location=torch.device('cuda:0')))

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #files[ta] for ta in range(t+1)
        model_without_ddp = model.module    
    

    files= [ 'coco_backpack_train_at', 'coco_book_train_at', 'coco_truck_train_at', 'coco_bowl_train_at', 'coco_bottle_train', 'coco_cup_train', 'coco_chair_train', 'coco_car_train'] #

    val_files= ['coco_backpack_validation', 'coco_book_validation', 'coco_truck_validation' , 'coco_bowl_validation', 'coco_bottle_validation']
    cor_files= ['coco_backpack_train_cr10', 'coco_book_train_cr10', 'coco_truck_train_cr10' , 'coco_bowl_train_cr10', 'coco_bottle_train_cr10']
    tok_root = ['/app/src/BLIP/dataset/coco_token_backpack_at','/app/src/BLIP/dataset/coco_token_book_at', '/app/src/BLIP/dataset/coco_token_truck_at' , '/app/src/BLIP/dataset/coco_token_bowl_at',  '/app/src/BLIP/dataset/coco_token_bottle', '/app/src/BLIP/dataset/coco_token_cup', '/app/src/BLIP/dataset/coco_token_chair',  '/app/src/BLIP/dataset/coco_token_car'  ] 
                   
    for t in range(args.num_tasks):
        if t>0:
            if t>0:
                update_annotation(model.module, '/app/src/BLIP/dataset/captions', files[t-1], val_files[t-1], '/app/src/BLIP/dataset/coco_train2014/train2014', tok_root[t-1], transform, args )
            print(files[t])
            datasets= vqa_dataset( transform, ann_root= '/app/src/BLIP/dataset/captions' ,img_root= '/app/src/BLIP/dataset/coco_train2014/train2014' , train_files=[files[t]] + [cor_files[ta] for ta in range(t)], split="train", task=0)
            if args.distributed:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()            
                sampler = DistributedSampler(datasets, shuffle=True,num_replicas=num_tasks , rank=global_rank)   
            else:
                samplers = [None, None, None]

            train_loader = utils.create_loader(datasets,samplers=sampler,batch_size=config['batch_size'],num_workers=2,is_trains=True, collate_fns=vqa_collate_fn)     
            best = 0
            best_epoch = 0
            
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])


            print("Start training")
            start_time = time.time()    
            for epoch in range(0, config['max_epoch']):
                if not args.evaluate:        
                    if args.distributed:
                        train_loader.sampler.set_epoch(epoch)

                    cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

                    train_stats = train(model, train_loader, optimizer, epoch, device) 

                if (epoch+1)%50 == 0:
                    torch.save(model.module.state_dict(),f'/app/src/BLIP/output/BLIP_AT_{t}_{epoch}.pt') 

                dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/app/src/BLIP/configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ratio', default=0.31, type=float)

    parser.add_argument('--num_tasks', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    rtpt = RTPT(name_initials='SP', experiment_name='Train IMAGE CAPTIONING', max_iterations=1000)
    rtpt.start()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
