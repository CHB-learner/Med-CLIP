import sys
from os.path import abspath, dirname

parent_dir = abspath(dirname(dirname(__file__)))
sys.path.append(parent_dir)

from clip import CLIP


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

    # avoid 0
    precisions = np.where(precisions == 0, 0.0000000001, precisions)
    recalls = np.where(recalls == 0, 0.0000000001, recalls)
    # 
    precisions = np.nan_to_num(precisions, nan=1e-10)
    recalls = np.nan_to_num(recalls, nan=1e-10)
    print('precisions',precisions)
    print('recalls',recalls)
    print('precisions+recalls',precisions + recalls)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    mean_f1_score = np.mean(f1_scores)
    return mean_f1_score



if __name__ == "__main__":
    clip = CLIP()
    #
    captions   = ['Primary Tumor','1','2','3','4','5','6','7',
               'Solid Tissue Normal',]
               
    number_one_class = 100
    csv = pd.read_csv('./1_estimate_chb/NCT_CRC_HE_100K/{}_test.csv'.format(number_one_class))   # 10_test_test.csv 20_test_test.csv all_test.csv
    tru_lable = []
    predict = []
    
    acc_dic = {caption: 0 for caption in captions}
    #
    sum = 0
    right = 0
    for index, row in csv.iterrows():
        sum = sum+1
        print("percentage: {:.4f}".format(index/len(csv['image'])))
        image_path = row['image']
        real_classification = row['classification']

        image = Image.open(image_path)
        probs = clip.detect_image(image, captions)
        # discribtion_match = captions[np.argmax(probs[0])]
        discribtion_match = random.choice(captions)
        #  print(discribtion_match)
        #  print(real_classification)
        # print("Label probs:", discribtion_match)
        tru_lable.append(real_classification)
        predict.append(discribtion_match)
# ------------------------------------------------
    print('predict over')
    # 
    confusion_mat = confusion_matrix(tru_lable, predict, labels=captions)
    
    print("confusion_mat:\n", confusion_mat)
    F1_Score = calculate_f1_score(confusion_mat)
    print(round(F1_Score,3))
    
    
    
    
    
    

