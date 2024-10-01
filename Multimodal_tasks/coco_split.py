# %matplotlib inline
# !pip install scikit-image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import requests

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/app/src/BLIP/dataset/coco_ann2014'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

names= ['person', 'sports', 'outdoor', 'vehicle', 'electronic' , 'animal', 'food']

for name in names:
    if name != 'person':
        print(name)
        catIds = coco.getCatIds(catNms=[name])
        # Get the corresponding image ids and images using loadImgs
        imgIds = coco.getImgIds(catIds=catIds)
        images = coco.loadImgs(imgIds)
        for im in tqdm(images):
            #if i > 42557:
            img_data = requests.get(im['coco_url']).content
            with open(f'/app/src/BLIP/dataset/coco_ann2014/coco_{name}/' + im['file_name'], 'wb') as handler:
                handler.write(img_data)

