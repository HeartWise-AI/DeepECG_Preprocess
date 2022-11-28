__author__ = 'alexis nolin-lapalme'
__email__ = 'alexis.nolin-lapalme@umontreal.ca'
__release__ = "0.1.0"

#utils
import fnmatch
import array
import base64
import re
import numpy as np
import pandas as pd
import argparse
import os

#argparse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Get argument',add_help=False)

    parser.add_argument("--data_path", metavar="data_path", type=str, help="Enter path to xml2dict output")
    parser.add_argument("--out_path", metavar="out_path", type=str, help="Output dir", default=".")
    parser.add_argument("--out_name", metavar="out_name", type=str, help="Output name", default="TinyGetWaveform.csv")
    parser.add_argument("--save", metavar="save", type=bool, help="Output dir", default=True)
    return parser


class TinyGetWaveform():
    def __init__(self, data:pd.DataFrame, out_name:str="TinyGetWaveform.csv",save:bool=True,save_path="/media/data1/anolin/ECG",save_npy:bool=True, compressed: bool=True):
        self.data = data
        self.save_npy = save_npy
        self.save_path = save_path
        self.compressed = compressed
        self.save = save
        self.out_name = out_name

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

        def longitudinal_substract(array_of_array_1:np.array,array_of_array_2:np.array, mult=1):
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
            for entry in range(13):
                lead_col_name = "{}_LeadData_{}_WaveFormData".format(wave,entry)
                if lead_col_name in self.data.columns:
                    lead_id= self.data["{}_LeadData_{}_LeadID".format(wave,entry)].dropna().values[0]
                    self.data['Lead_Wavform_{}_ID_{}'.format(wave[-1],lead_id)] = self.data[lead_col_name].map(transform_raw_lead)

            self.data['Lead_Wavform_{}_ID_III'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVR'.format(wave[-1])] = longitudinal_add(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVL'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values, mult=0.5)
            self.data['Lead_Wavform_{}_ID_aVF'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values, mult=0.5)

        if self.save == True:
            self.data.to_csv(os.path.join(self.save_path, self.out_name))

        return self.data


def main(args):
    df_ = pd.read_csv(args.data_path)
    TinyGetWaveform(data=df_,out_name=args.out_name,save=args.save,save_path=args.out_path).generate_dataset()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser('run the script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
