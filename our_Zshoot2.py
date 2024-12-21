# conda env:clip_for_test_chb



import sys
from os.path import abspath, dirname

parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)

from clip import CLIP

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score




# --------------------------------------------------------------------------------------------------
def fun_dataset(dataset_select):
    if dataset_select == 'Kather':
        captions=['MUCOSA',
                   'LYMPHO',
                   'DEBRIS',
                   'TUMOR',
                   'ADIPOSE',
                   'COMPLEX',
                   'STROMA',
                   'EMPTY']

    if dataset_select == 'DigestPath':
        captions=['Tumor', 'Solid Tissue Normal']

    if dataset_select == 'PanNuke':
        captions=['Uterus',
                   'Lung',
                   'Prostate',
                   'Cervix',
                   'Bile-duct',
                   'Skin',
                   'Testis',
                   'Thyroid',
                   'Ovarian',
                   'Kidney',
                   'Breast',
                   'Esophagus',
                   'Adrenal_gland',
                   'Stomach',
                   'Bladder',
                   'HeadNeck',
                   'Colon',
                   'Pancreatic',
                   'Liver']
 
    if dataset_select == 'WSSS4LUAD':
        captions=['Tumor', 'Solid Tissue Normal']
        
    if dataset_select == 'our':
        captions=['cancer_classification is Stomach adenocarcinoma',
                   'cancer_classification is Uveal Melanoma',
                   'cancer_classification is Kidney Chromophobe',
                   'cancer_classification is Uterine Carcinosarcoma',
                   'cancer_classification is Lung adenocarcinoma',
                   'cancer_classification is Kidney renal clear cell carcinoma',
                   'cancer_classification is Glioblastoma multiforme',
                   'cancer_classification is Colon adenocarcinoma',
                   'cancer_classification is Esophageal carcinoma ',
                   'cancer_classification is Pheochromocytoma and Paraganglioma',
                   'cancer_classification is Lymphoid Neoplasm Diffuse Large B-cell Lymphoma',
                   'cancer_classification is Pancreatic adenocarcinoma',
                   'cancer_classification is Prostate adenocarcinoma',
                   'cancer_classification is Thymoma',
                   'cancer_classification is Bladder Urothelial Carcinoma',
                   'cancer_classification is Uterine Corpus Endometrial Carcinoma',
                   'cancer_classification is Adrenocortical carcinoma',
                   'cancer_classification is Thyroid carcinoma',
                   'cancer_classification is Skin Cutaneous Melanoma',
                   'cancer_classification is Cervical squamous cell carcinoma and endocervical adenocarcinoma',
                   'cancer_classification is Rectum adenocarcinoma',
                   'cancer_classification is Kidney renal papillary cell carcinoma',
                   'cancer_classification is Testicular Germ Cell Tumors',
                   'cancer_classification is Mesothelioma',
                   'cancer_classification is Brain Lower Grade Glioma',
                   'cancer_classification is Sarcoma',
                   'cancer_classification is Lung squamous cell carcinoma',
                   'cancer_classification is Breast invasive carcinoma',
                   'cancer_classification is Liver hepatocellular carcinoma',
                   'cancer_classification is Cholangiocarcinoma',
                   'cancer_classification is Ovarian serous cystadenocarcinoma',
                   'cancer_classification is Head and Neck squamous cell carcinoma']
        
    return captions
# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    clip = CLIP()
    Best_F1 = [0,0,0]
    Best_ACC = [0,0,0]
    while True:
        for data_S in ['Kather','PanNuke','WSSS4LUAD','our']:
            dataset_select = data_S   # Kather  PanNuke WSSS4LUAD our
            captions = fun_dataset(dataset_select)
            
            
            
            ACC_LS = []
            ls_F1 = []
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            
            for number_one_class in [10,50,100]:
                dataset = '../test_csv/{}.csv'.format(dataset_select)
                csv_original = pd.read_csv(dataset)  
                
                
                tru_lable = []
                predict = []
                acc_dic = {caption: 0 for caption in captions}
                sum = 0
                right = 0
        
                csv = csv_original.sample(number_one_class)
                
                for index, row in csv.iterrows():
                    sum = sum+1
                    # print(number_one_class,"percentage: {:.4f}".format(sum/number_one_class))
                    
                    # image_path = row['file_name']
                    real_classification = row['classification']
                    if dataset_select == 'our':
                        image_path = row['file_name']
                    else:
                        image_path = '../test_img/{}/'.format(dataset_select)+row['file_name']
            
            
                    image = Image.open(image_path)
                    probs = clip.detect_image(image, captions)
                    discribtion_match = captions[np.argmax(probs[0])]
                    
                    tru_lable.append(real_classification)
                    predict.append(discribtion_match)
            
        
        
                    tru_lable.append(real_classification)
                    predict.append(discribtion_match)
                    if discribtion_match == real_classification:
                        right = right+1
                    # print(discribtion_match)
                    # print(real_classification)
                    # print()
            # ------------------------------------------------
                # print('predict over')
                # 
                confusion_mat = confusion_matrix(tru_lable, predict, labels=captions)
                
                # print("confusion_mat:\n", confusion_mat)
                F1_Score = metrics.f1_score(tru_lable, predict,  average='weighted') 
                # F1_Score = calculate_f1_score(confusion_mat)
                ACC_66 = accuracy_score(tru_lable, predict)
                
                
                #print('F1_score:',round(F1_Score,3))
                #print('ACC:',round(ACC_66,4))
                
                ls_F1.append(F1_Score)
                ACC_LS.append(round(ACC_66,4))
            
            # print('10 50 100')
            # print('F1:',ls_F1)
            # print('ACC:',ACC_LS)
            #-------------------------------------
            Best_F1 = [max(x, y) for x, y in zip(Best_F1, ls_F1)]
            Best_ACC = [max(x, y) for x, y in zip(Best_ACC, ACC_LS)]
            print()
            print('***************************************our model-{}***************************************'.format(dataset_select))
            print('10 50 100')
            print('Best_F1:',Best_F1)
            print('Best_ACC:',Best_ACC)
            print('****************************************')
            print()
        
    











