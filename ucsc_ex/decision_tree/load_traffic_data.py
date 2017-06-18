#!/usr/bin/python

import numpy as np
import pandas as pd

TRAFFIC_DATA_FILE = '/Users/manoj/my/course/ML/Code/ucsc_ex/decision_tree/Traffic_Sign_Data/Trafic_Sign_Data.csv'
TRAFFIC_TARGET_FILE = '/Users/manoj/my/course/ML/Code/ucsc_ex/decision_tree/Traffic_Sign_Data/Trafic_Sign_Target.csv'


def load_traffic(data_type='all', targets=[1, 2, 3, 4], testing_percent=25):
    data_df = pd.read_csv(TRAFFIC_DATA_FILE, header=None)
    target_df = pd.read_csv(TRAFFIC_TARGET_FILE, header=None)
    target_df.replace(['A', 'B', 'C', 'D'], [1, 2, 3, 4], inplace=True)
    target_df.columns = ['target']
    full_df = pd.concat([data_df, target_df], axis=1)
    full_df = full_df.loc[full_df['target'].isin(targets)]
    full_np = np.asarray(full_df)
    testing_split = int(((100.0-testing_percent)/100.0) * full_np.shape[0])
    if data_type is 'training':
        return full_np[0:testing_split,0:783], full_np[0:testing_split,784]
    elif data_type is 'testing':
        return full_np[testing_split:,0:783], full_np[testing_split:,784]
    else:
        return full_np[:, 0:783], full_np[:, 784]

if __name__ == '__main__':
    allimages, alllabels = load_traffic('training', [1,2])
    print (allimages.shape)
    print (alllabels.shape)
    allimages, alllabels = load_traffic('testing', [1, 2])
    print(allimages.shape)
    print(alllabels.shape)
    allimages, alllabels = load_traffic('all', [1, 2])
    print(allimages.shape)
    print(alllabels.shape)