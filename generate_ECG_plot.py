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
def plot_a_lead(row_num=200):

    #sample y to: 1 only allow 0.4 ms sampling rate 
    #             2 only plot 12 seconds of per lead


    lead_id_list = ['I','II','III', 'V1','V2','V3','V4','V5','V6', 'aVL','aVR','aVF']

    pannel_1_y = list()
    pannel_2_y = list()
    pannel_3_t = list()
    pannel_4_t = list()

    lead_dict = dict(zip(lead_id_list,[[] for i in range(len(lead_id_list))]))

    for lead in lead_id_list:
        y = data_set['Lead_Wavform_1_ID_{}'.format(lead)].iloc[row_num]
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

    fig, ax = plt.subplots(figsize=(40, 40))
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

    ax.xaxis.set_major_locator(ticker.MultipleLocator(125))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(ls='-', color='red', linewidth=4)
    ax.grid(which="minor", ls=':', color='red', linewidth=3)

    ax.axis([0-100, 2500+100, min(pannel_4_y)-10, max(pannel_1_y)+10])

    x = [pos for pos in range(0, len(pannel_1_y))]
    ax.plot(x,pannel_1_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_2_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_3_y,linewidth=3, color='#000000')
    ax.plot(x,pannel_4_y,linewidth=3, color='#000000')

    def replace_str_index(text,index=0,replacement=''):
        return '%s%s%s'%(text[:index],replacement,text[index+1:])

    def title_reshape(str):
        if len(str) > 100:
           num_patritions = round(len(str)/75)
           for i in range(1,num_patritions+1):
                for pos, entry in enumerate(list(str[75*i::])):
                    if entry == ' ':
                        replace_str_index(str,pos+(75*i),'\n')
                        break


    ax.set_title(title_reshape(re.sub("\s\s+" , " ",data_set["Diag"].iloc[row_num].replace("ECG anormal","").replace(";","\t"))), fontsize=20, y=0.96,  backgroundcolor='white')
    #plt.subplots_adjust(top=0.85)

    #add grid
    plt.tight_layout()

    #plt.savefig("/volume/validation_{}_{}_{}.jpg".format(data_set['RestingECG_PatientDemographics_PatientID'].iloc[row_num],data_set['RestingECG_TestDemographics_AcquisitionDate'].iloc[row_num], row_num))
plot_a_lead(row_num=49742)
