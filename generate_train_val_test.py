import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
from sklearn.model_selection import StratifiedShuffleSplit

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
    list_labels_string = list()
    list_labels_num = list()
    
    for pos,i in enumerate(data_set['Diag']):
        if 'Fibrillation auriculaire' in i and '**' not in i and 'ECG normal' not in i and '*Analyse impossible; aucun QRS décelable* ' not in i and 'Rythme sinusal normal' not in i:
            list_labels_string.append("FA")
            list_labels_num.append(0)

        elif 'Rythme sinusal normal' in i  and '**' not in i and '*Analyse impossible; aucun QRS décelable* ' not in i and 'Fibrillation auriculaire' not in i:
            list_labels_string.append("RS")
            list_labels_num.append(1)

        else:
            list_labels_string.append("other")
            list_labels_num.append(2)            



    print('number of FA: {}'.format(len([i for i in list_labels_string if i == 'FA'])))
    print('number of RS: {}'.format(len([i for i in list_labels_string if i == 'RS'])))
    print('number of other: {}'.format(len([i for i in list_labels_string if i == 'other'])))


    data_set['FA_RS_str'] = list_labels_string
    data_set['FA_RS_num'] = list_labels_num

    return data_set[data_set['FA_RS_str'] != 'other']

def make_binned_columns(df,binwidth,age_var):
    bined_columns = list()
    max_ = df[age_var].max()
    min_ = df[age_var].min()

    list_bins = list()
    last = min_
    while last < max_:
        list_bins.append([last,last+binwidth])
        last += binwidth

    #merge both extremities bins
    list_bins.append([list_bins[0][0],list_bins[1][1]])
    list_bins.append([list_bins[-2][0],list_bins[-1][1]])

    del list_bins[0]
    del list_bins[-3]
    del list_bins[0]
    del list_bins[-3]

    print(list_bins)

    list_new_list = list()
    for i in df[age_var]:
        checked = False 
        for min_,max_ in list_bins:
            if i < max_ and i >= min_:
                list_new_list.append((max_+min_)/2)
                checked = True

        if checked == False:
            list_new_list.append((list_bins[-1][0] + list_bins[-1][1])/2)
    


    df['bined_age'] = list_new_list
    return df

  
 def balanced_w_age_sex(df, split=[0.6,0.2,0.2], plot_distr=True, num_age_bin=10):
    
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

   #y=pd.get_dummies(df_bined[['bined_age','FA_RS_num',gender_var]])
   
    y = df_bined[['bined_age','FA_RS_str',gender_var]].values
    
    #sns.histplot(x=df['bined_age'].tolist())
    #plt.show()


    X_train, X_val_test, y_train, y_val_test = train_test_split(df_bined, y, test_size = 0.4, stratify=y)
    X_val,X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5,  stratify=y_val_test)

    #X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test, test_size=split[2], random_state=0, stratify=y_val_test)

    if plot_distr == True:
        sns.histplot(x=df[age_var].tolist(), binwidth=bin_width)
        sns.histplot(x=X_train[age_var].tolist(),binwidth=bin_width)
        sns.histplot(x=X_val[age_var].tolist(),binwidth=bin_width)
        sns.histplot(x=X_test[age_var].tolist(), binwidth=bin_width)
        plt.show()

    train_x_ = df2np(X_train)
    val_x_ = df2np(X_val)
    test_x_ = df2np(X_test)

    y_train = X_train['FA_RS_num'].values
    y_val = X_val['FA_RS_num'].values
    y_test = X_test['FA_RS_num'].values

    return train_x_,val_x_,test_x_, y_train,y_val,y_test



data = pd.read_csv("/media/data1/anolin/ECG/df_xml_corrected.csv")

train_x_,val_x_,test_x_, y_train,y_val,y_test= balanced_w_age_sex(data)

