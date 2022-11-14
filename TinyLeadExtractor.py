import fnmatch
import array
import base64
import re
pd.options.mode.chained_assignment = None
class TinyGetWaveform():
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

        for wave in ['RestingECG_Waveform_0','RestingECG_Waveform_1']:
            for entry in range(0,13):
                lead_col_name = "{}_LeadData_{}_WaveFormData".format(wave,entry)
                if lead_col_name in self.data.columns:
                    lead_id = self.data["{}_LeadData_{}_LeadID".format(wave,entry)].dropna().values[0]
                    self.data['Lead_Wavform_{}_ID_{}'.format(wave[-1],lead_id)] = self.data[lead_col_name].map(transform_raw_lead)

            self.data['Lead_Wavform_{}_ID_III'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVR'.format(wave[-1])] = longitudinal_add(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values)
            self.data['Lead_Wavform_{}_ID_aVL'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values, mult=0.5)
            self.data['Lead_Wavform_{}_ID_aVF'.format(wave[-1])] = longitudinal_substract(self.data['Lead_Wavform_{}_ID_II'.format(wave[-1])].values,self.data['Lead_Wavform_{}_ID_I'.format(wave[-1])].values, mult=0.5)
    
        return self.data
