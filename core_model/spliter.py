import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain
import warnings
import uuid
pd.set_option('display.max_columns', None)

dir = '/media/data1/ravram/DeepECG/ekg_waveforms_output/df_xml_2023_05_09_2004-to-june-2022_n_1572280_with_labelbox_no_duplicates.parquet'
data_ = pd.read_parquet(dir)
data_['RestingECG_PatientDemographics_PatientID'] = [str(i).zfill(7) for i in data_['RestingECG_PatientDemographics_PatientID'].tolist()]
data = data_.sort_values(by=['RestingECG_PatientDemographics_PatientID'])
data = data[data.annotated != -1]

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from: 
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size,train_size=train_size,
                                random_state=random_state, stratify=None, shuffle=shuffle)
    
    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    
    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )

def regenerate_final(final_data, dict_observations, original_dataframe):
    list_to_fill_final = list()
    index_final = final_data['RestingECG_PatientDemographics_PatientID'].tolist()

    for pos in tqdm(index_final):
        list_to_fill_final += dict_observations[pos]
    
    final_set = original_dataframe[original_dataframe.index.isin(list_to_fill_final)]

    return final_set

def regenerate_the_dataset(train_data, val_data, test_data, dict_observations, original_dataframe):
    list_to_fill_train = list()
    list_to_fill_val = list()
    list_to_fill_test = list()

    index_train = train_data['RestingECG_PatientDemographics_PatientID'].tolist()
    index_val = val_data['RestingECG_PatientDemographics_PatientID'].tolist()
    index_test = test_data['RestingECG_PatientDemographics_PatientID'].tolist()

    for pos in tqdm(index_train):
        list_to_fill_train += dict_observations[pos]

    for pos in tqdm(index_val):
        list_to_fill_val += dict_observations[pos]

    for pos in tqdm(index_test):
        list_to_fill_test += dict_observations[pos]

    train_set = original_dataframe[original_dataframe.index.isin(list_to_fill_train)]
    val_set = original_dataframe[original_dataframe.index.isin(list_to_fill_val)]
    test_set = original_dataframe[original_dataframe.index.isin(list_to_fill_test)]

    return train_set, val_set, test_set

def adjust_private_variables_to_balance(df, sex_groups=[35,65]):
    df['SEX_BIN'] = [0 if i == 'FEMALE' else 1 for i in df['RestingECG_PatientDemographics_Gender'].tolist()]
    df['AGE_BIN'] = [0 if i < sex_groups[0] else 2 if i > sex_groups[1] else 1 for i in df['RestingECG_PatientDemographics_PatientAge'].tolist()]

    return df

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def transform_(dataset, name, split=0):
    
    if not os.path.exists("/media/data1/anolin/split_smaller_for_ram/{}".format(name)) and name == 'final':
        os.mkdir("/media/data1/anolin/split_smaller_for_ram/{}".format(name))

    if not os.path.exists("/media/data1/anolin/split_smaller_for_ram/split_{}".format(split)) and name != 'final':
        os.mkdir("/media/data1/anolin/split_smaller_for_ram/{}".format(name))

    start_col = 280
    end_col = 357

    temp_data = list()
    list_to_remove = list()

    for path,labels in enumerate(tqdm(zip(dataset['npy_path'], dataset.index), total=len(dataset), desc='Generating X,Y data for final set')):


        if os.path.exists(path):

            t_ = np.squeeze(NormalizeData(np.load(path)).astype(np.float32))
            if np.isnan(np.sum(t_)):
                list_to_remove.append(labels)

            else:
                temp_data.append(t_)
        
        else: 
            list_to_remove.append(labels)       

    labels = dataset[~dataset.index.isin(list_to_remove)].iloc[:,start_col:end_col]

    if name == 'final':
        np.save("/media/data1/anolin/split_smaller_for_ram/{}/{}_Y.npy".format(name,name), labels.to_numpy())

        out_ = np.array(temp_data).astype(np.float32)
        print('final shape {}'.format(out_.shape))
        np.save("/media/data1/anolin/split_smaller_for_ram/{}/{}_X.npy".format(name,name), out_)

    else:
        np.save("/media/data1/anolin/split_smaller_for_ram/split_{}/{}_Y.npy".format(split,name), labels.to_numpy())

        out_ = np.array(temp_data).astype(np.float32)
        print('final shape {}'.format(out_.shape))
        np.save("/media/data1/anolin/split_smaller_for_ram/split_{}/{}_X.npy".format(split,name), out_)

import os
from termcolor import colored

def generate_balanced_split(df,seed_list=[420,1997,2023],sex_groups=[35,65],num_splits=3, notebook=False, outdir='/volume/core_model',**kwargs):

    #make sure the progress bar is shown correctly
    if notebook == True:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    assert len(seed_list) == num_splits

    try:
        os.mkdir(os.path.join(outdir, "data_split"))

    except:
        print(colored(os.path.join(outdir, "data_split"), 'red'), 'already exists')
        pass


    #clean the dataset
    df = adjust_private_variables_to_balance(df,sex_groups=sex_groups)

    #get one ECG per patient to prevent data leakage
    # generate a dict of patient id: [all index with an ECG from that patient]
    current_observation = df['RestingECG_PatientDemographics_PatientID'].tolist()[0]
    temp_list_locations = list()
    dict_observations = dict()
    for index, patient_id in tqdm(zip(df.index.tolist(),df['RestingECG_PatientDemographics_PatientID'].tolist()), desc='Generating patient:ECG dict', total=len(df['RestingECG_PatientDemographics_PatientID'].tolist())):
        if patient_id == current_observation:
            temp_list_locations.append(index)
        
        else:
            dict_observations.update({current_observation:temp_list_locations})
            current_observation = patient_id
            temp_list_locations = [index]

    # to prevent data leakage
    list_index = list()
    list_patient_id = list()

    dict_observations_2 = dict_observations.copy()
    for patient_id, index in dict_observations.items():
        list_index.append(index[0])
        list_patient_id.append(patient_id)

    #use this df in the split
    data_unique = df[df.index.isin(list_index)]
    
    start_col = 280
    end_col = 357
    to_balance = data_unique[data_unique.columns.tolist()[start_col:end_col+1] + ['SEX_BIN','AGE_BIN']]

    X_data, X_final, y_data, y_final = multilabel_train_test_split(data_unique,to_balance,stratify=to_balance, test_size=0.30,random_state=int(8.30**2))
    X_final = regenerate_final(X_final, dict_observations_2, df)
    X_final.to_csv(os.path.join("/media/data1/anolin/split_smaller_for_ram",'X_final.csv'))
    transform(X_final, 'final')

    to_balance = X_data[X_data.columns.tolist()[start_col:end_col+1] + ['SEX_BIN','AGE_BIN']]

    for pos in tqdm(range(num_splits), desc='Generating splits'):
        
        try:
            os.mkdir("/media/data1/anolin/split_smaller_for_ram/split_{}".format(pos))

        except:
            print(colored("/media/data1/anolin/split_smaller_for_ram/split_{}".format(pos), 'red'), 'already exists')
            pass      

        seed_everything(seed_list[pos])

        X_train, X_val_test, y_train, y_val_test = multilabel_train_test_split(X_data,to_balance,stratify=to_balance, test_size=0.15, random_state=seed_list[pos])
        X_val, X_test, y_val, y_test = multilabel_train_test_split(X_val_test,y_val_test,stratify=y_val_test, test_size=0.66, random_state=seed_list[pos])

        X_train, X_val, X_test = regenerate_the_dataset(X_train, X_val, X_test, dict_observations_2, df)

        #save the files
        X_train.to_csv(os.path.join("/media/data1/anolin/split_smaller_for_ram/split_{}".format(pos),'X_train.csv'))
        X_val.to_csv(os.path.join("/media/data1/anolin/split_smaller_for_ram/split_{}".format(pos),'X_val.csv'))
        X_test.to_csv(os.path.join("/media/data1/anolin/split_smaller_for_ram/split_{}".format(pos),'X_test.csv'))

        
        #generate the X data for each split, this is required to ensure no nan 
        transform(X_train, 'train', pos)
        transform(X_val, 'val', pos)
        transform(X_test, 'test', pos)


generate_balanced_split(data) 
