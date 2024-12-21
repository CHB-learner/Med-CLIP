# conda env:clip_for_test_chb

import sys
from os.path import abspath, dirname

parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)

# from clip import CLIP

import torch
import clip
from PIL import Image




from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def calculate_f1_score(confusion_mat):
    # 
    true_positives = np.diag(confusion_mat)
    false_positives = np.sum(confusion_mat, axis=0) - true_positives
    false_negatives = np.sum(confusion_mat, axis=1) - true_positives
    
    # 
    precisions = true_positives / (true_positives + false_positives)
    recalls = true_positives / (true_positives + false_negatives)

    # 
    # avoid 0
    precisions = np.where(precisions == 0, 0.0000000001, precisions)
    recalls = np.where(recalls == 0, 0.0000000001, recalls)
    # 
    precisions = np.nan_to_num(precisions, nan=1e-10)
    recalls = np.nan_to_num(recalls, nan=1e-10)
    
    print(precisions)
    print(recalls)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    mean_f1_score = np.mean(f1_scores)
    
    return mean_f1_score



if __name__ == "__main__":
    ls_F1 = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    CLIP, preprocess = clip.load("RN50x64", device=device) 
    
    for number_one_class in [10,50,100]:

        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    
        print(clip)
        
        
        # clip = CLIP()
        #
        captions   =  ['Colon adenocarcinoma','Head and Neck squamous cell carcinoma',
'Lung squamous cell carcinoma' ,'Uterine Corpus Endometrial Carcinoma',
'Ovarian serous cystadenocarcinoma' ,'Adrenocortical carcinoma',
'Thyroid carcinoma' ,'Breast invasive carcinoma' ,'Lung adenocarcinoma'
'Bladder Urothelial Carcinoma' ,'Prostate adenocarcinoma',
'Kidney renal clear cell carcinoma',"else"]
                   
        # /clip-pytorch/1_estimate_chb/NCT_CRC_HE_100K/10_test.csv
        csv = pd.read_csv('./clip-pytorch/1_estimate_chb/pubmed_set/{}_test.csv'.format(number_one_class))  # 10_test_test.csv 20_test_test.csv all_test.csv
        tru_lable = []
        predict = []
        
        acc_dic = {caption: 0 for caption in captions}
        #
        sum = 0
        right = 0
        text = clip.tokenize(captions).to(device)
        for index, row in csv.iterrows():
            sum = sum+1
            print("percentage: {:.4f}".format(index/len(csv['image'])))
            image_path = row['image']
            real_classification = row['classification']
    
    
            # 
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = CLIP.encode_image(image)
                text_features = CLIP.encode_text(text)
                
                logits_per_image, logits_per_text = CLIP(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
                # image = Image.open(image_path)
                # probs = clip.detect_image(image, captions)
                discribtion_match = captions[np.argmax(probs[0])]
            # print("Label probs:", discribtion_match)
            tru_lable.append(real_classification)
            predict.append(discribtion_match)
            if discribtion_match == real_classification:
                right = right+1
            print(discribtion_match)
            print(real_classification)
            print()
    # ------------------------------------------------
        print('predict over')
        # 
        confusion_mat = confusion_matrix(tru_lable, predict, labels=captions)
        
        print("confusion_mat:\n", confusion_mat)
        F1_Score = calculate_f1_score(confusion_mat)
        print('F1_score:',round(F1_Score,3))
        print('ACC:',round(right/sum,4))
        ls_F1.append(F1_Score)
    print('----------------------------')
    print('----------------------------')
    print('----------------------------')
    print(ls_F1)
    
    
'''
# conda env:clip_for_test_chb

# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./clip-pytorch/img/test.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

'''