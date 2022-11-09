from pylab import rcParams
import re

rcParams['figure.figsize'] = 30, 30
rcParams['xtick.bottom'] = rcParams['xtick.labelbottom'] = False
rcParams['ytick.left'] = rcParams['ytick.labelleft'] = False
def plot_a_lead(row_num=50):

    #sample y to: 1 only allow 0.4 ms sampling rate 
    #             2 only plot 12 seconds of per lead

    lead_id_list = ['I','II','III', 'V1','V2','V3','V4','V5','V6', 'aVL','aVR','aVF']

    pannel_1_y = list()
    pannel_2_y = list()
    pannel_3_t = list()
    pannel_4_t = list()

    lead_dict = dict(zip(lead_id_list,[[] for i in range(len(lead_id_list))]))

    #generate the lead activation
    activation = [0] *5 + [10] * 50 + [0] * 5


    for lead in lead_id_list:
        y = data_set['Lead_Wavform_1_ID_{}'.format(lead)].iloc[row_num]
        if len(y) == 5000:
            #resample at half the rate
            y = y[1::2] #sample half the points

        lead_dict[lead] = list(y)

    #generate the pannels

    pannel_1_y = activation + [i/4.88 for i in lead_dict['I'][0:625]] + [i/4.88 for i in lead_dict['aVR'][0:625]] + [i/4.88 for i in lead_dict['V1'][0:625]] + [i/4.88 for i in lead_dict['V4'][0:625]]
    pannel_2_y = activation + [i/4.88 for i in lead_dict['II'][0:625]] + [i/4.88 for i in lead_dict['aVL'][0:625]] + [i/4.88 for i in lead_dict['V2'][0:625]] + [i/4.88 for i in lead_dict['V5'][0:625]]
    pannel_3_y = activation + [i/4.88 for i in lead_dict['III'][0:625]] + [i/4.88 for i in lead_dict['aVF'][0:625]] + [i/4.88 for i in lead_dict['V3'][0:625]] + [i/4.88 for i in lead_dict['V6'][0:625]]
    pannel_4_y = activation + [i/4.88 for i in lead_dict['II']]

    minor_ticks = np.arange(0, 2500, 10)
    major_ticks = np.arange(0, 2500, 50)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    x = [pos for pos in range(0, len(pannel_1_y))]
    ax1.plot(x,pannel_1_y,linewidth=0.7, color='#000000')

    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)

    minor_ticks_y = np.arange(min(pannel_1_y), max(pannel_1_y), 1)
    major_ticks_y = np.arange(min(pannel_1_y), max(pannel_1_y), 5)

    ax1.set_yticks(major_ticks_y)
    ax1.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax1.grid(which='both')

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', color='#515151', linestyle=':')
    ax1.grid(which='major', color='#515151', linestyle='--')

    #add labels for leads
    ax1.vlines(60,10,-10, label='I')
    ax1.text(60, 11, 'I', fontsize=22)

    ax1.vlines(685,10,-10, label='aVR')
    ax1.text(685, 11, 'aVR', fontsize=22)

    ax1.vlines(1310,10,-10, label='V1')
    ax1.text(1310, 11, 'V1', fontsize=22)

    ax1.vlines(1935,10,-10, label='V4')
    ax1.text(1935, 11, 'V4', fontsize=22)

    ax2.plot(x,pannel_2_y, linewidth=.7,color='#000000')

    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax2.grid(which='both')

    # Or if you want different settings for the grids:
    ax2.grid(which='minor', color='#515151', linestyle=':')
    ax2.grid(which='major', color='#515151', linestyle='--')

    #add labels for leads
    ax2.vlines(60,10,-10, label='II')
    ax2.text(60, 11, 'II', fontsize=22)

    ax2.vlines(685,10,-10, label='aVL')
    ax2.text(685, 11, 'aVL', fontsize=22)

    ax2.vlines(1310,10,-10, label='V2')
    ax2.text(1310, 11, 'V2', fontsize=22)

    ax2.vlines(1935,10,-10, label='V5')
    ax2.text(1935, 11, 'V5', fontsize=22)

    minor_ticks_y = np.arange(min(pannel_2_y), max(pannel_2_y), 1)
    major_ticks_y = np.arange(min(pannel_2_y), max(pannel_2_y), 5)

    ax2.set_yticks(major_ticks_y)
    ax2.set_yticks(minor_ticks_y, minor=True)

    ax3.plot(x,pannel_3_y,linewidth=.7,color='#000000')

    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax3.grid(which='both')

    # Or if you want different settings for the grids:
    ax3.grid(which='minor', color='#515151', linestyle=':')
    ax3.grid(which='major', color='#000000', linestyle='--')

    #add labels for leads
    ax3.vlines(60,10,-10, label='III')
    ax3.text(60, 11, 'III', fontsize=22)

    ax3.vlines(685,10,-10, label='aVF')
    ax3.text(685, 11, 'aVF', fontsize=22)

    ax3.vlines(1310,10,-10, label='V3')
    ax3.text(1310, 11, 'V3', fontsize=22)

    ax3.vlines(1935,10,-10, label='V6')
    ax3.text(1935, 11, 'V6', fontsize=22)

    minor_ticks_y = np.arange(min(pannel_3_y), max(pannel_3_y), 1)
    major_ticks_y = np.arange(min(pannel_3_y), max(pannel_3_y), 5)

    ax3.set_yticks(major_ticks_y)
    ax3.set_yticks(minor_ticks_y, minor=True)

    ax4.plot(x,pannel_4_y,linewidth=.7,color='#000000')

    ax4.set_xticks(major_ticks)
    ax4.set_xticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax4.grid(which='both')

    # Or if you want different settings for the grids:
    ax4.grid(which='minor', color='#515151', linestyle=':')
    ax4.grid(which='major', color='#000000', linestyle='--')

    ax4.vlines(60,10,-10, label='II')
    ax4.text(60, 11, 'II', fontsize=22)

    minor_ticks_y = np.arange(min(pannel_4_y), max(pannel_4_y), 1)
    major_ticks_y = np.arange(min(pannel_4_y), max(pannel_4_y), 5)

    ax4.set_yticks(major_ticks_y)
    ax4.set_yticks(minor_ticks_y, minor=True)

    fig.suptitle(re.sub("\s\s+" , " ",data_set["Diag"].iloc[row_num].replace("ECG anormal","")), fontsize=12, y=0.99,  backgroundcolor='white')
    #plt.subplots_adjust(top=0.85)

    #add grid
    plt.tight_layout()

    # shift subplots down:
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("/volume/corrected_{}_{}.jpg".format(data_set['RestingECG_PatientDemographics_PatientID'].iloc[row_num],data_set['RestingECG_TestDemographics_AcquisitionDate'].iloc[row_num]))

