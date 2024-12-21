from PIL import Image
import os 
from clip import CLIP
import pandas as pd
import numpy as np
from utils.utils import (cvtColor, get_configs, letterbox_image,
                         preprocess_input)
                         
def cosine_similarity(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    similarity_normalized = 0.5 * (similarity + 1)
    return similarity_normalized





import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip import CLIP
from utils.dataloader import ClipDataset, dataset_collate
from utils.metrics import itm_eval


if __name__ == "__main__":
    im_name = []
    score_ls = []
    #------------------------------------------------------#
    #------------------------------------------------------#
    datasets_path               = "/heat_map_images/"
    datasets_val_json_path      = "./heat_map_images/TCGA-AA-A02H-01A-03-BS3.6d5f47a1-855b-457e-82b6-bd943c36fbe8.svs.json"
    datasets_val_json_path      = "./heat_map_images/TCGA-D8-A1JL-01A-01-TS1.9f253479-0b59-42c5-a7d9-00724cd03439.svs.json"
    SVS_name = datasets_val_json_path.split('/')[-1][:-5]
    print(SVS_name)
    
    
    batch_size                  = 1
    num_workers                 = 4

    # 
    model       = CLIP()

    # 
    val_lines   = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
    num_val     = len(val_lines)
    # 
    val_dataset = ClipDataset([model.config['input_resolution'], model.config['input_resolution']], val_lines, datasets_path, random = False)
    gen_val     = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=dataset_collate, sampler=None)

    # 
    i_features = []
    t_features = []
    with open(datasets_val_json_path, 'r') as f:
        data_ls = json.load(f)
    X = 0
    for iteration, batch in tqdm(enumerate(gen_val)):
        images, texts = batch
        file_name = data_ls[X]['image'].split('/')[-1]
        with torch.no_grad():
            if model.cuda:
                images  = images.cuda()
            images_feature, texts_feature = model.detect_image_for_eval(images, texts)
            images_feature = images_feature.cpu().numpy()
            texts_feature = texts_feature.cpu().numpy()
            
            
            
            
            score = cosine_similarity(images_feature,texts_feature)
            print('file_name',file_name,"score", score)
            im_name.append(file_name)
            score_ls.append(score)
            X = X+1
    dict_data={'name':im_name,'score':score_ls}
    df = pd.DataFrame(dict_data)
    csv_file = "./heat_map_images/{}.csv".format(SVS_name)
    df.to_csv(csv_file, index=False)
    
    
    
    
    
    