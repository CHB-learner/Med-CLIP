from sklearn.metrics import confusion_matrix
from PIL import Image
from clip import CLIP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


from PIL import Image
from clip import CLIP
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
    print('\n average_precisions',precisions)
    recalls = true_positives / (true_positives + false_negatives)
    print('\n average_recall',recalls)

    # 
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    mean_f1_score = np.mean(f1_scores)
    return mean_f1_score



if __name__ == "__main__":
    clip = CLIP()
    #
    captions   = ['tumor_stage is stage iii',
                   'tumor_stage is stage ii',
                   'tumor_stage is -999',
                   'tumor_stage is stage iv',
                   'tumor_stage is stage i',
                   'tumor_stage is stage iib',
                   'tumor_stage is stage iia',
                   'tumor_stage is stage iiia',
                   'tumor_stage is stage x',
                   'tumor_stage is stage ia',
                   'tumor_stage is stage iiic',
                   'tumor_stage is stage iiib',
                   'tumor_stage is stage ib',
                   'tumor_stage is stage ivb',
                   'tumor_stage is stage iva',
                   'tumor_stage is stage iic',
                   'tumor_stage is stage ivc',
                   'tumor_stage is stage 0',
                   'tumor_stage is i/ii nos',
                   'tumor_stage is is']
               
    number_one_class = 10
    csv = pd.read_csv('./1_estimate_chb/tumor_prdict/{}_tumor_test_test.csv'.format(number_one_class))  # 10_test_test.csv 20_test_test.csv all_test.csv
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
        real_classification = row['tumor_stage']

        image = Image.open(image_path)
        probs = clip.detect_image(image, captions)
        discribtion_match = captions[np.argmax(probs[0])]
        
        # print("Label probs:", discribtion_match)
        tru_lable.append(real_classification)
        predict.append(discribtion_match)
# ------------------------------------------------
    print('predict over \n')
    # 
    confusion_mat = confusion_matrix(tru_lable, predict, labels=captions)
    
    print("confusion_mat:\n", confusion_mat)
    
    # F1_score
    F1_Score = calculate_f1_score(confusion_mat)
    print('\n F1_score:',round(F1_Score,3))
    
    # accuracy
    accuracy = (confusion_mat.diagonal().sum()) / confusion_mat.sum()
    print('accuracy',accuracy)
    

    
    
    
    
    

