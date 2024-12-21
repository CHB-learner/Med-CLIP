# conda env:clip_for_test_chb



import sys
from os.path import abspath, dirname

parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)
import os
from clip import CLIP
import pandas as pd
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
    if dataset_select == 'days':
        captions =  [str(i) for i in range(1, 3001)]
    if dataset_select == 'stage':
        captions = ['stage one','stage two','stage three','stage four']
    if dataset_select == 'TNBC':
        my_list = list(range(100, 4900, 30))
        captions = list(map(str, my_list))
        
    return captions






# days and stage--------------------------------------------------------------
if __name__ == "__main__":
    samples = 10
    
    
    
    import pandas as pd
    clip = CLIP()
    predict_days_ls = []
    predict_stage_ls = []
    image_path_ls = []
    
    
    

    dataset_select = 'days'   # stage   days
    captions = fun_dataset(dataset_select)
    dataset = '../test_csv/our.csv'
    csv = pd.read_csv(dataset).sample(n=5000) 
    
    
    tru_lable = []
    predict = []
    acc_dic = {caption: 0 for caption in captions}
    sum = 0
    right = 0

    
    total = len(csv['file_name'].tolist())
    for index, row in csv.iterrows():
        sum = sum+1

        real_classification = row['classification']
        if dataset_select != 'our':
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
        print(discribtion_match)
        print('days',sum/total)
        # print(real_classification)
        predict_days_ls.append(discribtion_match)
        image_path_ls.append(image_path)
# ---------------------------------------------------
    dataset_select = 'stage'   # stage   days
    captions = fun_dataset(dataset_select)
    dataset = '../test_csv/our.csv'
    
    
    
    
    
    
    
    csv = pd.read_csv(dataset).sample(n=5000)   
    
    
    
    
    
    
    
    
    
    tru_lable = []
    predict = []
    acc_dic = {caption: 0 for caption in captions}
    sum = 0
    right = 0

    
    
    for index, row in csv.iterrows():
        sum = sum+1

        real_classification = row['classification']
        if dataset_select != 'our':
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
        print(discribtion_match)
        print('stage:',sum/total)
        # print(real_classification)
        predict_stage_ls.append(discribtion_match)       
# -----------------------------------------
        
        
        
        
    save = {'ID':csv['file_name'].tolist(),'image_path_ls':image_path_ls,'days':predict_days_ls,'stage':predict_stage_ls}
    df = pd.DataFrame(save)
    filename = './5000_days_stage.csv'
    # 
    df.to_csv(filename, index=False)
    print(filename)
        
'''
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    clip = CLIP()
    predict_days_ls = []
    predict_stage_ls = []
    image_path_ls = []
    
    
    dataset_select = 'TNBC'   # stage   days
    captions = fun_dataset(dataset_select)
    dataset = '../test_csv/TNBC_Clinical_info_slice_level.csv'
    csv = pd.read_csv(dataset)

    df_TNBC = pd.read_csv('../test_csv/TNBC_Clinical_info_slice_level.csv')
    
    # for i in os.listdir('../test_img/survive_analysis/'):
    P_ID_old = 9999
    XXXX = 9999
    for index, row in df_TNBC.iterrows():
        P_ID = row['Patient ID']
        print(P_ID)
        
        if P_ID_old == P_ID:
            XXXX = XXXX+1
            #print('6')
        else:
            XXXX = 1
            P_ID_old = row['Patient ID']
            #print('777')
            
        file_name = str(row['Patient ID'])+'_{}'.format(XXXX)
        image_path = '../test_img/survive_analysis/{}.png'.format(file_name)


        image = Image.open(image_path)
        probs = clip.detect_image(image, captions)
        discribtion_match = captions[np.argmax(probs[0])]
        
        print(discribtion_match)

        
        
        
        predict_days_ls.append(discribtion_match)
        # image_path_ls.append(i)
# ---------------------------------------------------
    df_TNBC['days_predict'] = predict_days_ls
    df_predict = df_TNBC
    df_predict.to_csv('./TNBC_days_predict.csv',index=False)
'''
    











