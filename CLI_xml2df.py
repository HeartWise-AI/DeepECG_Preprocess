__author__ = 'alexis nolin-lapalme'
__email__ = 'alexis.nolin-lapalme@umontreal.ca'


#utils
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from copy import copy
import xmltodict
from datetime import datetime
import argparse

#click arguments

def get_arguments():
    parser = argparse.ArgumentParser(description='Get argument',add_help=False)

    parser.add_argument("--xml_path", metavar="xml_path", type=str, help="Enter path to xml", default="/media/data1/muse_ge/ecg_retrospective")
    parser.add_argument("--out_path", metavar="out_path", type=str, help="Output dir", default="/volume")
    parser.add_argument("--verbose", metavar="verbose", type=bool, help="Do you want a progress bar?", default=True)
    parser.add_argument("--save", metavar="save", type=bool, help="Do you want to save [debug option]", default=True)
    return parser


class tinyxml2df():
    def __init__(self, in_path:str, out_path:str='/media/data1/anolin/ECG', verbose:bool=True, save:bool=True):
        self.path = in_path
        self.out_path = out_path
        self.verbose = verbose
        self.save = save

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

    def read2flatten(self):
        xml_dict_list = list()
        path_list = list()
        files_with_xml = [_ for _ in os.listdir(self.path) if _.endswith('.xml')]

        #iterate through all the files name verbose or not
        print("{} | Currently transforming {} xml files from dir {} into dict".format(datetime.now().strftime("%H:%M:%S"),len(files_with_xml),self.path))
        for pos,file_xml in enumerate(tqdm(files_with_xml) if self.verbose else files_with_xml): 
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
        df['xml_dir'] = files_with_xml
        
        if self.save == True:
            df.to_csv(os.path.join(self.out_path, "df_xml_{}_n_{}.csv".format(datetime.now().strftime("%Y_%m_%d"),df.shape[0])))
        return df

def main(args):
    tinyxml2df(args.xml_path,args.out_path,args.verbose,args.save).read2flatten()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some xml into a df',parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
