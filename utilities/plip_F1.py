# conda env:clip_for_test_chb

from PIL import Image
from transformers import CLIPProcessor, CLIPModel




import sys
from os.path import abspath, dirname

parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)

# from clip import CLIP

import torch
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
    # print(device)
    # CLIP, preprocess = clip.load("RN50x64", device=device) 
    
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    CLIP = CLIPModel.from_pretrained("./plip", local_files_only= True)
    preprocess = CLIPProcessor.from_pretrained("./plip", local_files_only= True)
    
    for number_one_class in [100,50,10]:
        dataset = './clip-pytorch/1_estimate_chb/classification_csv/{}_test.csv'.format(number_one_class)
        print(dataset)

        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    
        # print(CLIP)
        
        
        # clip = CLIP()
        #
        '''
        captions   =  ['cancer_classification is Colon adenocarcinoma',
                        'cancer_classification is Head and Neck squamous cell carcinoma',
                        'cancer_classification is Lung squamous cell carcinoma' ,
                        'cancer_classification is Uterine Corpus Endometrial Carcinoma',
                        'cancer_classification is Ovarian serous cystadenocarcinoma' ,
                        'cancer_classification is Adrenocortical carcinoma',
                        'cancer_classification is Thyroid carcinoma' ,
                        'cancer_classification is Breast invasive carcinoma' ,
                        'cancer_classification is Lung adenocarcinoma'
                        'cancer_classification is Bladder Urothelial Carcinoma' ,
                        'cancer_classification is Prostate adenocarcinoma',
                        'cancer_classification is Kidney renal clear cell carcinoma']
        '''
        #captions = ['Solid Tissue Normal', 'Primary Tumor','else']
        captions   = ['cancer_classification is Adrenocortical carcinoma',
               'cancer_classification is Bladder Urothelial Carcinoma',
               'cancer_classification is Breast invasive carcinoma',
               'cancer_classification is Cervical squamous cell carcinoma and endocervical adenocarcinoma',
               'cancer_classification is Cholangiocarcinoma',
               'cancer_classification is Colon adenocarcinoma',
               'cancer_classification is Lymphoid Neoplasm Diffuse Large B-cell Lymphoma',
               'cancer_classification is Esophageal carcinoma ',
               'cancer_classification is Glioblastoma multiforme',
               'cancer_classification is Head and Neck squamous cell carcinoma',
               'cancer_classification is Kidney Chromophobe',
               'cancer_classification is Kidney renal clear cell carcinoma',
               'cancer_classification is Kidney renal papillary cell carcinoma',
               'cancer_classification is Brain Lower Grade Glioma',
               'cancer_classification is Liver hepatocellular carcinoma',
               'cancer_classification is Lung adenocarcinoma',
               'cancer_classification is Lung squamous cell carcinoma',
               'cancer_classification is Mesothelioma',
               'cancer_classification is Ovarian serous cystadenocarcinoma',
               'cancer_classification is Pancreatic adenocarcinoma',
               'cancer_classification is Pheochromocytoma and Paraganglioma',
               'cancer_classification is Prostate adenocarcinoma',
               'cancer_classification is Rectum adenocarcinoma',
               'cancer_classification is Sarcoma',
               'cancer_classification is Skin Cutaneous Melanoma',
               'cancer_classification is Stomach adenocarcinoma',
               'cancer_classification is Testicular Germ Cell Tumors',
               'cancer_classification is Thyroid carcinoma',
               'cancer_classification is Thymoma',
               'cancer_classification is Uterine Corpus Endometrial Carcinoma',
               'cancer_classification is Uterine Carcinosarcoma',
               'cancer_classification is Uveal Melanoma']
                   
        # /clip-pytorch/1_estimate_chb/NCT_CRC_HE_100K/10_test.csv
        csv = pd.read_csv(dataset)  # 10_test_test.csv 20_test_test.csv all_test.csv
        tru_lable = []
        predict = []
        
        acc_dic = {caption: 0 for caption in captions}
        #
        sum = 0
        right = 0
        # text = clip.tokenize(captions).to(device)
        for index, row in csv.iterrows():
            sum = sum+1
            print(number_one_class,"percentage: {:.4f}".format(index/len(csv['image'])))
            image_path = row['image']
            real_classification = row['cancer_classification']
    
    
            # 
            image = Image.open(image_path)
            
            inputs = preprocess(text = captions,
                            images=image, return_tensors="pt", padding=True)

            outputs = CLIP(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  
            probs = probs.detach().numpy()
            
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
    











