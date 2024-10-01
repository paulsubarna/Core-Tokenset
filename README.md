# Core Tokensets for Data-efficient Sequential Training of Transformers
This is the official pytorch repository of Core Tokensets. 
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Table of Contents
- [Installation](#installation)
- Sequential Training with Core Tokensets
	- [Sequential Image Classification Task](#Sequential-Image-Classification-Task)
	- [Sequential VQA task](#Sequential-VQA-task)
-  [Sequential Image Captioning task](#Sequential-Image-Captioning-task)
	- [Bibtex](#bibtex)

# Installation 
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

### Image Classification
1. Download ImageNet 1k from the original website and set 'image_root' in configs.
2. Set the root to Image classification
3. Set the memory root
4. To sequential train ViT with core tokensets, run:
 <pre>python -m torch.distributed.run --nproc_per_node=2\ 
Multimodal_tasks/VIT_{'choose core tokenset approach: atman, gradlrp,gradcam,rollout'}.py \
--num-epochs 15 --num-task 5 --num-class 20 --pix-ratio 0.1 --train-value 1 --mem-ratio 0.1 \
--ID 0.9 --drop-val False --random False --seed 5 --num-batches 64  </pre>

### Sequential VQA task:
1. Download VQA v2 dataset, CLEVR dataset, and Visual Genome dataset from the original websites, and set 'vqa_root' ,'clevr_root' and 'vg_root' in configs/vqa.yaml.
2. Set the path of the memory buffer where you would like to store the core tokens in configs/vqa.yaml
3. To sequentially train BLIP on three different VQA dataset, first set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 \ 
		Multimodal_tasks/train_vqa_{'choose core tokenset approach: atman or gradlrp'}.py </pre> 
4. To evaluate the finetuned BLIP model, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=2 eval_vqa.py --evaluate</pre> 

### Sequential-Image-Captioning-task:
1. Download COCO datasets from the original websites, and set 'image_root' in configs/caption_coco.yaml.
2. Set the path of the memory buffer where you would like to store the core tokens in configs/coco.yaml
3. To train the model sequentially with core tokenset, first set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth", then run: <pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py </pre> 
<pre>python -m torch.distributed.run --nproc_per_node=8 Multimodal_tasks/train_caption.py --evaluate</pre> 
4. To evaluate the finetuned BLIP model, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_caption_{'choose core tokenset approach: atman or gradlrp'}.py </pre> 



