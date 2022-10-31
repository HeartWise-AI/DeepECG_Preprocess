import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from copy import copy
import xmltodict

class tinyxml2df():
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
        xml_dict_list = []
        path_list = []
        files_with_xml = [_ for _ in os.listdir(self.path) if _.endswith('.xml')]

        #iterate through all the files name verbose or not
        print(
            f'{datetime.now().strftime("%H:%M:%S")} | Currently transforming {len(files_with_xml)} xml files from dir {self.path} into dict'
        )

        for file_xml in tqdm(files_with_xml) if verbose else files_with_xml:
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

        if save:
            df.to_csv(os.path.join(output_dir, "df_xml.csv"))
        return df
