import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def df2np(df:pd.DataFrame):
    arr_df = list()
    with tqdm(total=df.shape[0]) as pbar:
        for pos in range(df.shape[0]):
            if len(df.iloc[pos]['Lead_Wavform_1_ID_I']) == 5000:
                arr_ = np.array([df.iloc[pos]['Lead_Wavform_1_ID_I'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_II'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_III'][1::2],
                                df.iloc[pos]['Lead_Wavform_1_ID_V1'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_V2'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_V3'][1::2],
                                df.iloc[pos]['Lead_Wavform_1_ID_V4'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_V5'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_V6'][1::2],
                                df.iloc[pos]['Lead_Wavform_1_ID_aVL'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_aVR'][1::2],df.iloc[pos]['Lead_Wavform_1_ID_aVF'][1::2]])
            else:
                arr_ = np.array([df.iloc[pos]['Lead_Wavform_1_ID_I'],df.iloc[pos]['Lead_Wavform_1_ID_II'],df.iloc[pos]['Lead_Wavform_1_ID_III'],
                            df.iloc[pos]['Lead_Wavform_1_ID_V1'],df.iloc[pos]['Lead_Wavform_1_ID_V2'],df.iloc[pos]['Lead_Wavform_1_ID_V3'],
                            df.iloc[pos]['Lead_Wavform_1_ID_V4'],df.iloc[pos]['Lead_Wavform_1_ID_V5'],df.iloc[pos]['Lead_Wavform_1_ID_V6'],
                            df.iloc[pos]['Lead_Wavform_1_ID_aVL'],df.iloc[pos]['Lead_Wavform_1_ID_aVR'],df.iloc[pos]['Lead_Wavform_1_ID_aVF']])      
            arr_df.append(arr_)
            pbar.update(1)

    arr_final = np.array(arr_df)
    return arr_final
  
def add_labels_FA_RS(data_set):
    list_fa = list()
    for pos,i in enumerate(data_set['Diag']):
        if 'Fibrillation auriculaire' in i and '**' not in i and 'ECG normal' not in i and '*Analyse impossible; aucun QRS décelable* ' not in i and 'Rythme sinusal normal' not in i:
            list_fa.append(pos)

    list_rs = list()
    for pos,i in enumerate(data_set['Diag']):
        if 'Rythme sinusal normal' in i  and '**' not in i and 'ECG normal' not in i and '*Analyse impossible; aucun QRS décelable* ' not in i and 'Fibrillation auriculaire' not in i:
            list_rs.append(pos)

    list_labels_string = ['FA'] * data_set.shape[0]
    list_labels_num = [0] * data_set.shape[0]

    for i in list_rs:
        list_labels_string[i] == 'RS'
        list_labels_num[i] == 1

    print('number of FA: {}'.format(len(list_fa)))
    print('number of RS: {}'.format(len(list_rs)))

    data_set['FA_RS_str'] = list_labels_string
    data_set['FA_RS_num'] = list_labels_num

    return data_set
 

def make_binned_columns(df,binwidth,age_var):
    bined_columns = list()
    max_ = df[age_var].max()
    min_ = df[age_var].min()

    list_bins = list()
    last = min_
    while last < max_:
        list_bins.append([last,last+binwidth])
        last += binwidth

    list_new_list = list()
    pbar = tqdm(total=len(df[age_var].values)) 
    for i in df[age_var]:
        for min_,max_ in list_bins:
            if i < max_ and i >= min_:
                list_new_list.append((max_+min_)/2)
                pbar.update(1)
    pbar.close()

    df['bined_age'] = list_new_list
    return df
  
 def balanced_w_age_sex(df, split=[0.6,0.2,0.2], plot_distr=True, num_age_bin=20, seed=3):
    
    age_var = 'RestingECG_PatientDemographics_PatientAge'
    gender_var = 'RestingECG_PatientDemographics_Gender'

    #add the labels for the task
    df = add_labels_FA_RS(df)

    #get bins
    min_age = df[age_var].min()
    max_age = df[age_var].max()

    bin_width = (max_age - min_age)/num_age_bin

    df_bined = make_binned_columns(df,bin_width,age_var)

    df_bined['combined'] = df_bined['bined_age'].astype(str) + '_' + df_bined[age_var].astype(str) + '_' + df_bined['FA_RS_str'].astype(str)

    X = np.random.uniform(size=df_bined.shape[0])
    y=df_bined[['bined_age','FA_RS_str',gender_var]].values

    X_train, X_val_test, y_train, y_val_test = train_test_split(df_bined,y, test_size=split[1]+split[2], random_state=seed, stratify=y)


    X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test, test_size=split[1]+split[2], random_state=seed, stratify=y_val_test)

    if plot_distr == True:
        sns.histplot(x=df[age_var].tolist(), binwidth=bin_width)
        sns.histplot(x=X_train[age_var].tolist(),binwidth=bin_width)
        sns.histplot(x=X_val[age_var].tolist(),binwidth=bin_width)
        sns.histplot(x=X_test[age_var].tolist(), binwidth=bin_width)
        plt.show()

    train_x = df2np(X_train)
    val_x = df2np(X_val)
    test_x = df2np(X_test)

 
    train_y = X_train['FA_RS_num'].values()
    val_y = X_val['FA_RS_num'].values()
    test_y = X_test['FA_RS_num'].values()

    return train_x,val_x,test_x,train_y,val_y,test_y


data = pd.read_csv("/media/data1/anolin/ECG/df_xml_corrected.csv")

train_x,val_x,test_x,train_y,val_y,test_y = balanced_w_age_sex(data)

