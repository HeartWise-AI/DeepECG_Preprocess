from pylab import rcParams
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
import matplotlib.ticker as ticker

rcParams['figure.figsize'] = 30, 30
rcParams['xtick.bottom'] = rcParams['xtick.labelbottom'] = False
rcParams['ytick.left'] = rcParams['ytick.labelleft'] = False


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

def flatten(input_node: dict, key_: str = '', output_dict: dict = {}):
    if isinstance(input_node, dict):
        for key, val in input_node.items():
            new_key = f"{key_}_{key}" if key_ else f"{key}"
            flatten(val, new_key, output_dict)
    elif isinstance(input_node, list):
        for idx, item in enumerate(input_node):
            flatten(item, f"{key_}_{idx}", output_dict)
    else:
        output_dict[key_] = input_node
    return output_dict

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
    if not np.isnan(np.min(array_of_array_1)) and not np.isnan(np.min(array_of_array_2)):
        return [np.subtract(array_of_array_1[i],mult*array_of_array_2[i]) for i in range(len(array_of_array_1))]
    else:
        return np.nan

def longitudinal_add(array_of_array_1:np.array,array_of_array_2:np.array, mult=-0.5):
    if not np.isnan(np.min(array_of_array_1)) and not np.isnan(np.min(array_of_array_2)):
        return [np.add(array_of_array_1[i],array_of_array_2[i])*mult for i in range(len(array_of_array_1))]
    else:
        return np.nan

def plot_from_xml(xml_file):

    with open(os.path.join(xml_file), 'r') as xml:

        ECG_data_nested = xmltodict.parse(xml.read())
        ECG_data_flatten = flatten(ECG_data_nested)

        lead_id = [i for i in ECG_data_flatten.keys() if ('RestingECG_Waveform_1_' in i and "LeadID" in i)]
        lead_infromation = [i for i in ECG_data_flatten.keys() if ('RestingECG_Waveform_1_' in i and "_WaveFormData" in i)]
        ECG_data_flatten_only_leads = {'{}_{}'.format(info_,ECG_data_flatten[id_]):transform_raw_lead(ECG_data_flatten[info_]) for id_,info_ in zip(lead_id,lead_infromation)}

        ECG_data_flatten_only_leads = {k.split('_')[-1]:v for k,v in ECG_data_flatten_only_leads.items()}

        #generate oterh leads
        ECG_data_flatten_only_leads.update({'III':np.array(longitudinal_substract(ECG_data_flatten_only_leads['II'],ECG_data_flatten_only_leads['I']))})
        ECG_data_flatten_only_leads.update({'aVR':np.array(longitudinal_add(ECG_data_flatten_only_leads['I'],ECG_data_flatten_only_leads['II']))})
        ECG_data_flatten_only_leads.update({'aVL':np.array(longitudinal_substract(ECG_data_flatten_only_leads['I'],ECG_data_flatten_only_leads['II'],mult=0.5))})
        ECG_data_flatten_only_leads.update({'aVF':np.array(longitudinal_substract(ECG_data_flatten_only_leads['II'],ECG_data_flatten_only_leads['I'],mult=0.5))})

    # plot code

    return ECG_data_flatten_only_leads


def plot_a_lead(dict_):

    #sample y to: 1 only allow 0.4 ms sampling rate 
    #             2 only plot 12 seconds of per lead


    lead_id_list = ['I','II','III', 'V1','V2','V3','V4','V5','V6', 'aVL','aVR','aVF']

    pannel_1_y = list()
    pannel_2_y = list()
    pannel_3_t = list()
    pannel_4_t = list()

    lead_dict = dict(zip(lead_id_list,[[] for i in range(len(lead_id_list))]))

    for lead in lead_id_list:
        y = dict_[lead]
        if len(y) == 5000:
            #resample at half the rate
            y = y[1::2] #sample half the points

        lead_dict[lead] = list(y)

    #generate the lead activation
    activation = [0] *5 + [10] * 50 + [0] * 5

    #generate the pannels
    pannel_1_y = [i + 50 for i in activation] + [((i*4.88)/100) + 50 for i in lead_dict['I'][60:625]] + [((i*4.88)/100) + 50 for i in lead_dict['aVR'][0:625]] + [((i*4.88)/100) + 50 for i in lead_dict['V1'][0:625]] + [((i*4.88)/100) + 50 for i in lead_dict['V4'][0:625]]
    pannel_2_y = [i + 15 for i in activation] + [((i*4.88)/100) + 15 for i in lead_dict['II'][60:625]] + [((i*4.88)/100) + 15  for i in lead_dict['aVL'][0:625]] + [((i*4.88)/100) + 15  for i in lead_dict['V2'][0:625]] + [((i*4.88)/100) + 15  for i in lead_dict['V5'][0:625]]
    pannel_3_y = [i - 15 for i in activation] + [((i*4.88)/100) - 15 for i in lead_dict['III'][60:625]] + [((i*4.88)/100) - 15 for i in lead_dict['aVF'][0:625]] + [((i*4.88)/100) - 15 for i in lead_dict['V3'][0:625]] + [((i*4.88)/100) - 15 for i in lead_dict['V6'][0:625]]
    pannel_4_y = [i - 50 for i in activation] + [((i*4.88)/100) - 50 for i in lead_dict['II'][60::]]

    fig, ax = plt.subplots(figsize=(40, 20))
    ax.minorticks_on()
 
    ax.vlines(60,-10,-20, label='III', linewidth=4)
    ax.text(60, -10, 'III', fontsize=44)

    ax.vlines(625,-10,-20, label='aVF', linewidth=4)
    ax.text(625, -10, 'aVF', fontsize=44)

    ax.vlines(1250,-10,-20, label='V3', linewidth=4)
    ax.text(1250, -10, 'V3', fontsize=44)

    ax.vlines(1875,-10,-20, label='V6', linewidth=4)
    ax.text(1875, -10, 'V6', fontsize=44)
    
    ax.vlines(60,10,20, label='II', linewidth=4)
    ax.text(60, 20, 'II', fontsize=44)

    ax.vlines(625,10,20, label='aVL', linewidth=4)
    ax.text(625, 20, 'aVL', fontsize=44)

    ax.vlines(1250,10,20, label='V2', linewidth=4)
    ax.text(1250, 20, 'V2', fontsize=44)

    ax.vlines(1875,10,20, label='V5', linewidth=4)
    ax.text(1875, 20, 'V5', fontsize=44)

    ax.vlines(60,45,55, label='I', linewidth=4)
    ax.text(60, 55, 'I', fontsize=44)

    ax.vlines(625,45,55, label='aVR', linewidth=4)
    ax.text(625, 55, 'aVR', fontsize=44)

    ax.vlines(1250,45,55, label='V1', linewidth=4)
    ax.text(1250, 55, 'V1', fontsize=44)

    ax.vlines(1875,45,55, label='V4', linewidth=4)
    ax.text(1875, 55, 'V4', fontsize=44)

    ax.vlines(60,-55,-45, label='II', linewidth=4)
    ax.text(60, -45, 'II', fontsize=44)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(ls='-', color='red', linewidth=1.2)
    ax.grid(which="minor", ls=':', color='red', linewidth=1)

    ax.axis([0-100, 2500+100, min(pannel_4_y)-10, max(pannel_1_y)+10])

    x = [pos for pos in range(0, len(pannel_1_y))]
    ax.plot(x,pannel_1_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_2_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_3_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_4_y,linewidth=3, color='#000000')

    def replace_str_index(text,index=0,replacement=''):
        return '%s%s%s'%(text[:index],replacement,text[index+1:])

    def title_reshape(string):
        if len(string) > 200:
           num_patritions = round(len(string)/150)
           for i in range(1,num_patritions+1):
                for pos, entry in enumerate(list(string[150*i::])):
                    if entry == ' ':
                        string = replace_str_index(string,pos + (150*i), '\n')
                        break
        return string

    #add grid
    plt.tight_layout()
    
    
# Example
# import os
# dir_ = "/media/data1/muse_ge/ecg_retrospective"
# xml_dir = "MUSE_20221108_191339_22000.xml"
# plot_a_lead(plot_from_xml(os.path.join(dir_,xml_dir)))
