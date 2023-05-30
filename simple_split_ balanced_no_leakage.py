data_dir = "/media/data1/ravram/DeepECG/ekg_waveforms_output/df_xml_2023_05_09_2004-to-june-2022_n_1572280_with_labelbox_no_duplicates.parquet"
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
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

def regenerate_the_dataset(train_data, val_data, test_data, dict_observations, original_dataframe):
    list_to_fill_train = list()
    list_to_fill_val = list()
    list_to_fill_test = list()

    index_train = train_data.index.tolist()
    index_val = val_data.index.tolist()
    index_test = test_data.index.tolist()

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

sns.set(rc={'figure.figsize':(11.7,6.27)})

def adjust_private_variables_to_balance(df, sex_groups=[35,65]):
    df['SEX_BIN'] = [0 if i == 'FEMALE' else 1 for i in df['RestingECG_PatientDemographics_Gender'].tolist()]
    df['AGE_BIN'] = [0 if i < sex_groups[0] else 2 if i > sex_groups[1] else 1 for i in df['RestingECG_PatientDemographics_PatientAge'].tolist()]

    return df


def generate_split_no_leakage(data_dir, split=[0.7,0.1,0.2], age_groupe = [35,65], plot=True, interpolate=True):

    #load data 
    data = pd.read_parquet(data_dir)
    #remove unnanotated entries
    data = data[data.annotated == 0]
    data['RestingECG_PatientDemographics_PatientID'] = [str(i).zfill(7) for i in data['RestingECG_PatientDemographics_PatientID'].tolist()]
    data = data.sort_values(by=['RestingECG_PatientDemographics_PatientID'])

    # generate an dict of patient id: [all index with an ECG from that patient]
    current_observation = data['RestingECG_PatientDemographics_PatientID'].tolist()[0]
    temp_list_locations = list()
    dict_observations = dict()
    for index, patient_id in tqdm(zip(data.index.tolist(),data['RestingECG_PatientDemographics_PatientID'].tolist()), total=len(data['RestingECG_PatientDemographics_PatientID'].tolist())):
        if patient_id == current_observation:
            temp_list_locations.append(index)
        
        else:
            dict_observations.update({current_observation:temp_list_locations})
            current_observation = patient_id
            temp_list_locations = [index]

    # to prevent data leakage
    list_index = list()
    list_patient_id = list()
    for patient_id, index in dict_observations.items():
        list_index.append(index[0])
        list_patient_id.append(patient_id)

    data_unique = data[data.index.isin(list_index)]

    dict_temp_ensure_alignement = dict(zip(list_index,list_patient_id))
    data_unique.index = [dict_temp_ensure_alignement[i] for i in data_unique.index.tolist()]

    data_unique = adjust_private_variables_to_balance(data_unique, sex_groups=[35,65])
    data = adjust_private_variables_to_balance(data, sex_groups=[35,65])

    if plot == True:
        fig, axs = plt.subplots(1, 3)

        sns.countplot(data['RestingECG_PatientDemographics_Gender'].tolist(),ax=axs[0])
        sns.histplot(data['RestingECG_PatientDemographics_PatientAge'].tolist(),binwidth=10,ax=axs[1])
        sns.countplot(data['AGE_BIN'].tolist(),ax=axs[2])

        axs[1].axvline(x=age_groupe[0], color='k', linestyle='--', alpha=0.7)
        axs[1].axvline(x=age_groupe[1], color='k', linestyle='--', alpha=0.7)
        fig.suptitle("Distribution for all dataset")
        axs[0].title.set_text('Countplot gender')
        axs[1].title.set_text('Histplot age')
        axs[2].title.set_text('Countplot bined age')
        fig.tight_layout()
        plt.show()


        fig, axs = plt.subplots(1, 3)

        sns.countplot(data_unique['RestingECG_PatientDemographics_Gender'].tolist(),ax=axs[0])
        sns.histplot(data_unique['RestingECG_PatientDemographics_PatientAge'].tolist(),binwidth=10,ax=axs[1])
        sns.countplot(data_unique['AGE_BIN'].tolist(),ax=axs[2])

        axs[1].axvline(x=age_groupe[0], color='k', linestyle='--', alpha=0.7)
        axs[1].axvline(x=age_groupe[1], color='k', linestyle='--', alpha=0.7)
        fig.suptitle("Distribution for unique dataset")
        axs[0].title.set_text('Countplot gender')
        axs[1].title.set_text('Histplot age')
        axs[2].title.set_text('Countplot bined age')
        fig.tight_layout()
        plt.show()


    start_col = 280
    end_col = 357

    y_target = data_unique.iloc[:,start_col:end_col+1]
    to_balance = data_unique[data_unique.columns.tolist()[start_col:end_col+1] + ['SEX_BIN','AGE_BIN']]

    X_train, X_test, y_train, y_test = multilabel_train_test_split(y_target,to_balance,stratify=to_balance, test_size=1-split[0])
    X_val, X_test, y_val, y_test = multilabel_train_test_split(X_test,y_test,stratify=y_test, test_size=split[2]/(split[1]+split[2]))       

    if plot == True:

        fig, axs = plt.subplots(3, 2)

        sns.countplot(y_train['SEX_BIN'].tolist(),ax=axs[0,0])
        sns.countplot(y_train['AGE_BIN'].tolist(),ax=axs[0,1])

        sns.countplot(y_val['SEX_BIN'].tolist(),ax=axs[1,0])
        sns.countplot(y_val['AGE_BIN'].tolist(),ax=axs[1,1])

        sns.countplot(y_test['SEX_BIN'].tolist(),ax=axs[2,0])
        sns.countplot(y_test['AGE_BIN'].tolist(),ax=axs[2,1])

        fig.suptitle("Distribution for splits")

        axs[0,0].title.set_text('Sex bins train')
        axs[0,1].title.set_text('Age bins train')

        axs[1,0].title.set_text('Sex bins val')
        axs[1,1].title.set_text('Age bins val')

        axs[2,0].title.set_text('Sex bins test')
        axs[2,1].title.set_text('Age bins test')

        fig.tight_layout()
        plt.show()

    train_set, val_set, test_set = regenerate_the_dataset(X_train, X_val, X_test, dict_observations, data)


    train_set_X = generate_12_lead_df(train_set, interpolate)
    val_set_X = generate_12_lead_df(val_set, interpolate)
    test_set_X = generate_12_lead_df(test_set, interpolate)

    train_set_Y = train_set.iloc[:,start_col:end_col+1]
    val_set_Y = val_set.iloc[:,start_col:end_col+1]
    test_set_Y = test_set.iloc[:,start_col:end_col+1]

    return train_set_X,val_set_X,test_set_X,train_set_Y,val_set_Y,test_set_Y
