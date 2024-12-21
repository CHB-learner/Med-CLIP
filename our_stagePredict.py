# conda env:clip_chb



import sys
from os.path import abspath, dirname
import os
parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)

from clip import CLIP
from collections import Counter
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
    if dataset_select == 'our_stagePredict':
        captions=['stage one',
                   'stage two',
                   'stage three',
                   'stage four']
        
    return captions
# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    clip = CLIP()
    
    dataset_select = 'our_stagePredict'   
    captions = fun_dataset(dataset_select)
    dataset = '../test_csv/{}.csv'.format(dataset_select) # our_stagePredict.csv
    csv_original = pd.read_csv(dataset)  


    sa = len(csv_original)
    
    csv = csv_original #.sample(sa)



    #while True:
    dic_sum = {}
    dic_acc = {}
    for i in list(set(csv['cancer_classification'].tolist() )):
        dic_sum[i] = 0
        dic_acc[i] = 0
    num_log = 0
    for index, row in csv.iterrows():
        num_log = num_log + 1
        print('{}/{}'.format(num_log,sa))
        
        dir_png = row['svs_dir']
        cancer_class_real = row['cancer_classification']
        cancer_real_stage = row['cancer_staging']
        dic_sum[cancer_class_real] = dic_sum[cancer_class_real] + 1
        
        predict_result = []
        for j in os.listdir('../dataset/Open_GDC/{}/'.format(dir_png)):
            image_path = '../dataset/Open_GDC/{}/{}'.format(dir_png,j)
            image = Image.open(image_path)
            probs = clip.detect_image(image, captions)
            discribtion_match = captions[np.argmax(probs[0])]
            predict_result.append(discribtion_match)
        word_counts = Counter(predict_result)
        most_common_word, count = word_counts.most_common(1)[0]
        if most_common_word == cancer_real_stage:
            print('right')
            dic_acc[cancer_class_real] = dic_acc[cancer_class_real] + 1
    print('--------')
    print(dic_sum)
    print(dic_acc)
    
    
    
    # 
    accuracy_dict = {}
    for category in dic_sum.keys():
        total = dic_sum[category]
        correct = dic_acc.get(category, 0)
        accuracy = correct / total if total > 0 else 0
        accuracy_dict[category] = accuracy
    print('----accuracy_dict----')
    print(accuracy_dict)
    out = pd.DataFrame(list(accuracy_dict.items()), columns=['Cancer Type', 'Accuracy'])
    out.to_csv('./tumor_stage_predict.csv',index=False)
    
    
    # 
    total_counts = dic_sum
    total_samples = sum(total_counts.values())
    weighted_accuracy = sum(dic_acc.get(category, 0) / total_counts[category] * total_counts[category] / total_samples
                            for category in total_counts.keys())
    print('')
    print('')
    print('')
    print('')
    print('')
    print("Weighted Accuracy:", weighted_accuracy)



'''
# ------------------------------------------------

    confusion_mat = confusion_matrix(tru_lable, predict, labels=captions)
    F1_Score = metrics.f1_score(tru_lable, predict,  average='weighted') 
    ACC_Score = accuracy_score(tru_lable, predict)
    print('F1_score:',round(F1_Score,3))
    print('ACC:',round(ACC_66,4))
        
    
    #-------------------------------------
    
    print()
    print('***************************************our model-{}***************************************'.format(dataset_select))
    print('****************************************')
    print()
'''
    











