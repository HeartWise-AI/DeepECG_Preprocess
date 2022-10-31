import fnmatch
import array
import base64
import re

class GetWaveform():
    #okay, so there is an effect of nesting here boiz and girls
    #so we want to allow the user to first get either waveform by GetWaveformObject.waveform{1/2}
    #and then let the wonderful user decide to chose the lead volate/labels with GetWaveformObject.waveform{1/2}.LeadID{I,II,III...}
    #now let's dive in
    def __init__(self, data:pd.DataFrame, augmented_leads: bool=True, modify_df: bool=True):
        self.data = data

        def transform_raw_lead(lead_data):
            if lead_data != np.nan:
                lead_b64 = base64.b64decode(lead_data)
                return np.array(array.array("h", lead_b64))
            else:
                return lead_data

        #get the waveforms with a regex matching in a set comprehension
        self.RestingECG_Waveform_0 = self.GetLeads1
        self.RestingECG_Waveform_1 = self.GetLeads2

        self.lead_data_pairs = dict()
        for wave in ['RestingECG_Waveform_0','RestingECG_Waveform_1']:
            temp_dict = dict()
            for entry in range(1,13):
                try:
                    col_name = "{}_LeadData_{}_WaveFormData".format(wave,entry)
                    temp_dict.update({self.data[col_name].values[0]:tuple(zip(self.data[col_name].map(transform_raw_lead).values(),\
                        self.data['Original_Diag'].map(transform_raw_lead).values(),self.data['Diag'].map(transform_raw_lead).values()))})
                except:
                    pass
            self.lead_data_pairs.update({wave:temp_dict})

        if modify_df == True:
            for old_col in [col_lead for col_lead in self.data.columns.tolist() if re.match(re.compile('RestingECG_Waveform_._LeadData_.WaveFormData'),col_lead)]:
                self.data['{}_Transformed'.format(old_col)] = self.data[old_col].map(transform_raw_lead)

    class GetLeads1():
        def __init__(self):
            for lead_id,lead_data in GetWaveform.lead_data_pairs['RestingECG_Waveform_0'].items():
                setattr(self, lead_id, lead_data)

            #special leads
            def longitudinal_substract(list_1:np.array, list_2:np.array, multiplicator=0):
                return np.array([np.subtract(list_1[i],multiplicator*list_2[i]) for i in list_1])

            def longitudinal_add(list_1:np.array, list_2:np.array, multiplicator=-0.5):
                return np.array([np.add(list_1[i],list_2[i])*multiplicator for i in list_1])

            if 'II' in GetWaveform.lead_data_pairs['RestingECG_Waveform_0'].keys() and "I" in GetWaveform.lead_data_pairs['RestingECG_Waveform_0'].keys():
                self.III = (longitudinal_substract(self.II[0], self.I[0]),self.I[1],self.I[2])
                self.aVR = (longitudinal_add(self.I[0], self.II[0]),self.I[1],self.I[2])
                self.aVL = (longitudinal_substract(self.I[0], self.II[0],multiplicator=0.5),self.I[1],self.I[2])
                self.aVF = (longitudinal_substract(self.II[0], self.I[0],multiplicator=0.5),self.I[1],self.I[2])

        def __call__(self):
            return GetWaveform.lead_data_pairs['RestingECG_Waveform_0']

    class GetLeads2():
        def __init__(self):
            for lead_id,lead_data in GetWaveform.lead_data_pairs['RestingECG_Waveform_1'].items():
                setattr(self, lead_id, lead_data)

        def __call__(self):
            return GetWaveform.lead_data_pairs['RestingECG_Waveform_1']

    def __str__(self):
        #this is here to yield a message
        #when nthe object is called for debugging
        return "Robert want's a object descriptor here?"
