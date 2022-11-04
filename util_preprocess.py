import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from copy import copy
import xmltodict
import fnmatch
import array
import base64
import re

class tinyxml2df():
    """
    reads the xml directory and makes a csv
    
    TO USE: tinyxml2df(<dir name>).read2flatten()
    """
    def __init__(self, path:str):
        self.path = path
        
    def flatten(self,input_node: dict, key_: str = '', output_dict: dict = {}):
        if isinstance(input_node, dict):
            for key, val in input_node.items():
                new_key = f"{key_}_{key}" if key_ else f"{key}"
                self.flatten(val, new_key, output_dict)
        elif isinstance(input_node, list):
            for idx, item in enumerate(input_node):
                self.flatten(item, f"{key_}_{idx}", output_dict)
        else:
            output_dict[key_] = input_node
        return output_dict

    def fusediagcols(self, df:pd.DataFrame):
        df["Original_Diag"] = ""
        df["Diag"] = ""

        for col in [_ for _ in df.columns if 'RestingECG_OriginalDiagnosis_DiagnosisStatement' in _]:
            df["Original_Diag"]= df["Original_Diag"].str.cat(df[col].fillna('').copy(), sep =" ")
        for col in [_ for _ in df.columns if 'RestingECG_Diagnosis_DiagnosisStatement' in _]:
            df["Diag"]= df["Diag"].str.cat(df[col].fillna('').copy(), sep =" ")
        
        df["Diag"] = df["Diag"].str.replace(r'ENDSLINE', '')
        df["Diag"] = df["Diag"].str.replace(r'USERINSERT', '')

        df["Original_Diag"] = df["Original_Diag"].str.replace(r'ENDSLINE', '')
        df["Original_Diag"] = df["Original_Diag"].str.replace(r'USERINSERT', '')

        return df

    def check_abnoramlity(self, data:pd.DataFrame):
        warn = ["Analyse impossible", "ECG anormal"]
        list_abnormality = [0] * data.shape[0]
        for pos,entry in enumerate(data['Original_Diag'].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1

        for pos,entry in enumerate(data['Diag'].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1       
        data['warnings'] = list_abnormality
        return data

    def read2flatten(self, verbose: bool=True, output_dir: str='/media/data1/anolin/ECG', save: bool=True):
        xml_dict_list = list()
        path_list = list()
        files_with_xml = [_ for _ in os.listdir(self.path) if _.endswith('.xml')]

        #iterate through all the files name verbose or not
        print("{} | Currently transforming {} xml files from dir {} into dict".format(datetime.now().strftime("%H:%M:%S"),len(files_with_xml),self.path))
        for pos,file_xml in enumerate(tqdm(files_with_xml) if verbose else files_with_xml): 
            with open(os.path.join(self.path,file_xml), 'r') as xml:
                path_list.append(os.path.join(self.path,file_xml))
                #load
                ECG_data_nested = xmltodict.parse(xml.read())
                #flatten
                ECG_data_flatten = self.flatten(ECG_data_nested)
                #append to the list
                xml_dict_list.append(ECG_data_flatten.copy())

        df = self.fusediagcols(pd.DataFrame(xml_dict_list))
        df = self.check_abnoramlity(df)
        
        if save == True:
            df.to_csv(os.path.join(output_dir, "df_xml.csv"))
        return df
      
class TinyGetWaveform():
  """
  Takes care of managing the unprocessed numpy arrays and makes them into normal voltages
  """
    def __init__(self, data:pd.DataFrame, save_npy:bool=True, save_path="/media/data1/anolin/ECG", compressed: bool=True):
        self.data = data
        self.save_npy = save_npy
        self.save_path = save_path
        self.compressed = compressed
    def generate_dataset(self):  

        def transform_raw_lead(lead_data):
            if pd.isna(lead_data) == False:
                try:
                    lead_b64 = base64.b64decode(lead_data)
                    return np.array(array.array("h", lead_b64))
                except:
                    raise ValueError(pd.isna(lead_data))
                    
            else:
                return lead_data

        def longitudinal_substract(array_of_array_1:np.array,array_of_array_2:np.array, mult=0):
            if array_of_array_1 != np.nan and array_of_array_2 != np.nan:
                return [np.subtract(array_of_array_1[i],mult*array_of_array_2[i]) for i in range(len(array_of_array_1))]
            else:
                return np.nan

        def longitudinal_add(array_of_array_1:np.array,array_of_array_2:np.array, mult=-0.5):
            if array_of_array_1 != np.nan and array_of_array_2 != np.nan:
                return [np.add(array_of_array_1[i],array_of_array_2[i])*mult for i in range(len(array_of_array_1))]
            else:
                return np.nan

        #conventional boyz
        for wave in ['RestingECG_Waveform_0','RestingECG_Waveform_1']:
            for entry in range(0,13):
                lead_col_name = "{}_LeadData_{}_WaveFormData".format(wave,entry)
                if lead_col_name in self.data.columns:
                    lead_id= self.data["{}_LeadData_{}_LeadID".format(wave,entry)].dropna().values[0]
                    self.data['Lead_Wavform_{}_ID_{}'.format(wave[-1],lead_id)] = self.data[lead_col_name].map(transform_raw_lead)
        
            self.data['Lead_Wavform_{}_ID_III'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVR'.format(wave[-1])] = longitudinal_add(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVL'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values, mult=0.5)
            self.data['Lead_Wavform_{}_ID_aVF'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values, mult=0.5)
        """
        if self.save_npy == True:
            #create main dir
            try:
                os.mkdir(os.path.join(self.save_path, "numpy_ecg"))
            except:
                print("{} already exists".format(os.path.join(self.save_path, "numpy_ecg")))
            #create a patient-specific direcotry to save
            #wveform and label vectors
            list_numpy_paths = list()
            for xml in  self.data['xml_path'].values:
                try:
                    list_numpy_paths.append(os.path.join(self.save_path, "numpy_ecg",xml.split('/')[-1].split('.xml')[0]))
                    os.mkdir(os.path.join(self.save_path, "numpy_ecg",xml.split('/')[-1].split('.xml')[0]))
                except:
                    pass
           
            for col in self.data.columns:
                #save lead data
                if 'Lead_Wavform_' in col:
                    for pos,patient_info_entry in self.data[col].values:
                        loc = os.path.join(self.save_path, "numpy_ecg",self.data['xml_path'].iloc[pos].split('/')[-1].split('.xml')[0])
                        #I offer to compress
                        if self.compressed == True:
                            np.savez_compressed(os.path.join(loc,"{}.npz".format(col)), patient_info_entry)
                        else:
                            np.save(os.path.join(loc,"{}.npy".format(col)), patient_info_entry)        
                
                #save label vectors
                #diag
                for pos, entry in self.data['Original_Diag'].values:
                    loc = os.path.join(self.save_path, "numpy_ecg",self.data['xml_path'].iloc[pos].split('/')[-1].split('.xml')[0])
                    if self.compressed == True:
                        np.savez_compressed(os.path.join(loc,"Original_Diagnosis.npz"), entry)
                    else:
                        np.save(os.path.join(loc,"Original_Diagnosis.npy"), entry)
                #original_diag
                for pos, entry in self.data['Diag'].values:
                    loc = os.path.join(self.save_path, "numpy_ecg",self.data['xml_path'].iloc[pos].split('/')[-1].split('.xml')[0])
                    if self.compressed == True:
                        np.savez_compressed(os.path.join(loc,"Diagnosis.npz"), entry)
                    else:
                        np.save(os.path.join(loc,"Diagnosis.npy"), entry)
                self.data['numpy_path'] = list_numpy_paths
            """
        return self.data
      
def df2np(df:pd.DataFrame):
    
    arr_df = list()
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
    list_bins.append([list_bins[-3][0],list_bins[-2][1]])

    del list_bins[0]
    del list_bins[-3]
    del list_bins[0]
    del list_bins[-3]

    print('the binranges are: {}'.format(list_bins))

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
        plt.title("Distiribution of Ages")
        total_patch = mpatches.Patch(color='#5799c6', label='Total Distribution')
        train_patch = mpatches.Patch(color='#d4853b', label='Train distribution')
        val_patch = mpatches.Patch(color='#569830', label='Val distribution')
        test_patch = mpatches.Patch(color='#b5442a', label='Test distribution')
        plt.legend(handles=[total_patch,train_patch,val_patch,test_patch])
        plt.show()

    train_x_ = df2np(X_train)
    val_x_ = df2np(X_val)
    test_x_ = df2np(X_test)

    y_train = X_train['FA_RS_num'].values
    y_val = X_val['FA_RS_num'].values
    y_test = X_test['FA_RS_num'].values

    return train_x_,val_x_,test_x_, y_train,y_val,y_test
      
