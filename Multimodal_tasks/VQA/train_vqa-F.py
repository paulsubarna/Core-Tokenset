
import argparse
import os
#import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from rtpt import RTPT
import torch
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from patchify import patchify, unpatchify
from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset
os.chdir('/app/src/BLIP')
import matplotlib.pyplot as plt
from data.vqa_dataset import vqa_dataset, vqa_collate_fn, vqa_collate_fn_val
import utils, train_vqa

from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor, 
                                   ToPILImage)

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        loss = model(image, question, answer, train=True, n=n, weights=weights)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id, _, _,_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id)       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})   

    return result


def create_tokens(img):

    patches = patchify(img, (16,16,3), step=16)
    patches= patches.squeeze(2)
    patches = patches.reshape(196,16,16,3)
    return patches


def update_annotation(model, ann_root, task, transform, args):
    
    annotation = json.load(open(os.path.join(ann_root,'vqa_train-HC.json'),'r'))
    #annotate= annotation[10*task-1: 10*task]
    #model= model.cpu()
    for i in tqdm(range(50000*(task-1), 50000*task)):
        question= annotation[i]['question']
        id= annotation[i]['question_id']
        answer= annotation[i]['answer']
        img_root = '/app/src/BLIP/dataset/coco_train2014'
        image =Image.open(os.path.join(img_root, annotation[i]['image'])).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).cuda()
        
        attn= model.forward_chefer(image, question, answer , weights=1/len(annotation[i]['answer']))
        attn= attn.reshape(196)
        #t= get_attribution(image, model, use_thresholding =False, apply_transforms= False, apply_pixratio= False)  #1. get the attribution map from the lrp
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

        patches= patches.reshape(14,14,1,16,16,3)                 #unpatchify back to the image 
        img= unpatchify(patches, (224,224,3))  
        img= torch.from_numpy(img)
        img= img.permute(2,0,1)
        #plt.imsave(f'/app/src/BLIP/dataset/token_image/coco_{id}.png', img)
        torch.save(img, f'/app/src/BLIP/dataset/token_image/coco_{id}.pt')
        annotation[i]['image_memory']= f'coco_{id}.pt'
        annotation[i]['memory']= 'yes'
        del image
        torch.cuda.empty_cache()
    
    with open('/app/src/BLIP/dataset/vqa/vqa_train-HC.json', "w") as jsonFile:
        json.dump(annotation, jsonFile)


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    #datasets = create_dataset('vqa', config) 
    transform = transforms.Compose([
            transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
    #config['pretrained']
    print("Creating model")
    model = blip_vqa(pretrained=None, image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model.load_state_dict(torch.load('/app/src/BLIP/output/VQA/BLIPCLV_0_19.pt'))
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  
    start_time = time.time()
    for t in range(args.num_task):  
        if t >2:
            continue
        else:
            
            if t ==0:
                datasets= vqa_dataset( transform, ann_root= config['annot_root'] , vqa_root= config['vqa_root'], mem_root=config['vqa_mem_root'], vg_root=config['vg_root'], train_files=[config['train_files'][t]], split="train", task=t)
            elif t ==1:
                datasets= vqa_dataset( transform, ann_root= config['annot_root'] , vqa_root= config['vqa_root'],  vg_root=config['vg_root'], mem_root=config['vqa_mem_root'], doc_root= config['clevr_root'] , train_files=config['train_files'][:t+1], split="train", task=t)
            else:
                datasets= vqa_dataset( transform, ann_root= config['annot_root'] , vqa_root= config['vqa_root'],  vg_root=config['vg_root'], mem_root=config['vqa_mem_root'], doc_root= config['clevr_mem_root'], train_files=config['train_files'][:t+1], split="train", task=t)
            if args.distributed:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()            
                #samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)    
                sampler = DistributedSampler(datasets, shuffle=True,num_replicas=num_tasks , rank=global_rank)
            else:
                sampler = None
            #train_loader= []


            train_loader=utils.create_loader(datasets,samplers=sampler,batch_size=config['batch_size_train'],num_workers=4,is_trains=True, collate_fns=vqa_collate_fn)

            optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

            best = 0
            best_epoch = 0 

            print("Start training")


            for epoch in range(0, config['max_epoch']):
                if not args.evaluate:        
                    if args.distributed:
                        train_loader.sampler.set_epoch(epoch)

                    cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

                    train_stats = train(model, train_loader, optimizer, epoch, device) 

                else:         
                    break        

                if utils.is_main_process():     
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 'epoch': epoch,
                                }                
                    with open(os.path.join(args.output_dir, f"log_{t}.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                        

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    #torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{t}_%02d.pth'%epoch))  

                if (epoch+1)%10 == 0:
                    torch.save(model.module.state_dict(),f'/app/src/BLIP/output/VQA/BLIPF_{t+2}_{epoch}_{args.ratio}.pt') 
                dist.barrier()         

            #vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
            #result_file = save_result(vqa_result, args.result_dir, f'vqa_result_{t}')  

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-task', default=1, type=int)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    rtpt = RTPT(name_initials='SP', experiment_name='Train VQA', max_iterations=1000)
    rtpt.start()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
