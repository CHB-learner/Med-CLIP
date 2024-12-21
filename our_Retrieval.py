import json
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from clip import CLIP
from utils.dataloader import ClipDataset, dataset_collate
from utils.metrics import itm_eval
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import cv2
import re

def preprocess_input(x):
    x /= 255
    x -= np.array((0.48145466, 0.4578275, 0.40821073))
    x /= np.array((0.26862954, 0.26130258, 0.27577711))
    return x

def pre_caption(caption):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    
    return caption
if __name__ == "__main__":
    for dddd in ['PanNuke'  ,'our']:  #'Kather',  
        print(dddd)
        dataset_select = dddd  #Kather  PanNuke  our
        #------------------------------------------------------#
        #------------------------------------------------------#
        datasets_path               = "datasets/"
        datasets_val_json_path      = "datasets/en_val.json"
        batch_size                  = 32
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
        
        V_em = []
        Text_em = []
        # READ FROM OUT
        # for iteration, batch in tqdm(enumerate(gen_val)):
        
        
        csv_data = pd.read_csv('../test_csv/{}.csv'.format(dataset_select))
        if dddd == 'PanNuke':
            csv = csv_data
        else:
            csv = csv_data.sample(n=5000)
        
        transform = ToTensor()
        for index, row in csv.iterrows():
            print(index,'/',len(csv))
            # images, texts = batch
            if dataset_select == 'our':
                image_path = row['file_name']
            else:
                image_path = '../test_img/{}/'.format(dataset_select)+row['file_name']
            # images = Image.open(image_path)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (224, 224))
            images = torch.from_numpy(resized_image).float() 
            images = preprocess_input(images)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            
            
            
            text = row['classification']
            text = pre_caption(text)
            
            with torch.no_grad():
                if model.cuda:
                    images  = images.cuda()
                    print(images.shape)
                    images = images.unsqueeze(0)
                    images = images.permute(0, 3, 1, 2)
                    print(images.shape)
                
                images_feature, _ = model.detect_image_for_eval(images, texts=None)
                i_features.append(images_feature)
                V_em.append(images_feature.detach().cpu().numpy())
                
    
                _, texts_feature = model.detect_image_for_eval(images=None, texts=text)
                t_features.append(texts_feature)
                Text_em.append(texts_feature.detach().cpu().numpy())
    
    #    texts       = gen_val.dataset.text
    #    num_text    = len(texts)
    #    for i in tqdm(range(0, num_text, batch_size)):
    #        text = texts[i: min(num_text, i + batch_size)]
    #        with torch.no_grad():
    #            _, texts_feature = model.detect_image_for_eval(images=None, texts=text)
    #            t_features.append(texts_feature)
    #            Text_em.append(texts_feature.detach().cpu().numpy())
    
        print('len(V_em)==len(Text_em)',(len(V_em)==len(Text_em)))
        with open('../Retrieva_pkl/V_OUR_{}.pkl'.format(dataset_select), 'wb') as f:
            pickle.dump(V_em, f)
        f.close()
                
                
        with open('../Retrieva_pkl/L_OUR_{}.pkl'.format(dataset_select), 'wb') as f:
            pickle.dump(Text_em, f)
            print('len(Text_em):',len(Text_em))
        f.close()
        
        
        print('-------------------')
        print('OK-------------------')
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
    