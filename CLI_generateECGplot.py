__author__ = "alexis nolin-lapalme"
__email__ = "alexis.nolin-lapalme@umontreal.ca"
__release__ = "0.1.0"

import argparse
import os
import re

import bwr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tqdm
from pylab import rcParams
from scipy.signal import butter, filtfilt


# argparse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description="Get argument", add_help=False)

    parser.add_argument(
        "--data_path", metavar="data_path", type=str, help="Enter path to xml2dict output"
    )
    parser.add_argument(
        "--out_path", metavar="out_path", type=str, help="Output dir", default="."
    )
    parser.add_argument(
        "--verbose", metavar="verbose", type=str, help="Want progress bar?", default=True
    )

    return parser


def plot_a_lead(data_set, save_path, row_num=1):
    rcParams["figure.figsize"] = 30, 30
    rcParams["xtick.bottom"] = rcParams["xtick.labelbottom"] = False
    rcParams["ytick.left"] = rcParams["ytick.labelleft"] = False

    # sample y to: 1 only allow 0.4 ms sampling rate
    #             2 only plot 12 seconds of per lead

    T = 2.0  # Sample Period
    fs = 10.0  # sample rate, Hz
    cutoff = (
        2  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    )
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2  # sin wave can be approx represented as quadratic
    n = int(T * fs)  # total number of samples

    def butter_lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)
        return y

    lead_id_list = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVL", "aVR", "aVF"]

    pannel_1_y = []
    pannel_2_y = []
    pannel_3_t = []
    pannel_4_t = []

    lead_dict = dict(zip(lead_id_list, [[] for i in range(len(lead_id_list))]))

    for lead in lead_id_list:
        y = data_set[f"Lead_Wavform_1_ID_{lead}"].iloc[row_num]
        if len(y) == 5000:
            # resample at half the rate
            y = y[1::2]  # sample half the points

        lead_dict[lead] = list(bwr.bwr(butter_lowpass_filter(y, cutoff, fs, order))[1])

    # generate the lead activation
    activation = [0] * 5 + [10] * 50 + [0] * 5

    # generate the pannels
    pannel_1_y = (
        [i + 50 for i in activation]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["I"][60:625]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["aVR"][:625]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["V1"][:625]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["V4"][:625]]
    )

    pannel_2_y = (
        [i + 15 for i in activation]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["II"][60:625]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["aVL"][:625]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["V2"][:625]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["V5"][:625]]
    )

    pannel_3_y = (
        [i - 15 for i in activation]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["III"][60:625]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["aVF"][:625]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["V3"][:625]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["V6"][:625]]
    )

    pannel_4_y = [i - 50 for i in activation] + [
        ((i * 4.88) / 100) - 50 for i in lead_dict["II"][60::]
    ]

    fig, ax = plt.subplots(figsize=(40, 20))
    ax.minorticks_on()

    ax.vlines(60, -10, -20, label="III", linewidth=4)
    ax.text(60, -10, "III", fontsize=44)

    ax.vlines(625, -10, -20, label="aVF", linewidth=4)
    ax.text(625, -10, "aVF", fontsize=44)

    ax.vlines(1250, -10, -20, label="V3", linewidth=4)
    ax.text(1250, -10, "V3", fontsize=44)

    ax.vlines(1875, -10, -20, label="V6", linewidth=4)
    ax.text(1875, -10, "V6", fontsize=44)

    ax.vlines(60, 10, 20, label="II", linewidth=4)
    ax.text(60, 20, "II", fontsize=44)

    ax.vlines(625, 10, 20, label="aVL", linewidth=4)
    ax.text(625, 20, "aVL", fontsize=44)

    ax.vlines(1250, 10, 20, label="V2", linewidth=4)
    ax.text(1250, 20, "V2", fontsize=44)

    ax.vlines(1875, 10, 20, label="V5", linewidth=4)
    ax.text(1875, 20, "V5", fontsize=44)

    ax.vlines(60, 45, 55, label="I", linewidth=4)
    ax.text(60, 55, "I", fontsize=44)

    ax.vlines(625, 45, 55, label="aVR", linewidth=4)
    ax.text(625, 55, "aVR", fontsize=44)

    ax.vlines(1250, 45, 55, label="V1", linewidth=4)
    ax.text(1250, 55, "V1", fontsize=44)

    ax.vlines(1875, 45, 55, label="V4", linewidth=4)
    ax.text(1875, 55, "V4", fontsize=44)

    ax.vlines(60, -55, -45, label="II", linewidth=4)
    ax.text(60, -45, "II", fontsize=44)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(ls="-", color="red", linewidth=1.2)
    ax.grid(which="minor", ls=":", color="red", linewidth=1)

    ax.axis([0 - 100, 2500 + 100, min(pannel_4_y) - 10, max(pannel_1_y) + 10])

    x = list(range(len(pannel_1_y)))
    ax.plot(x, pannel_1_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_2_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_3_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_4_y, linewidth=3, color="#000000")

    def replace_str_index(text, index=0, replacement=""):
        return f"{text[:index]}{replacement}{text[index + 1:]}"

    def title_reshape(string):
        if len(string) > 200:
            num_patritions = round(len(string) / 150)
            for i in range(1, num_patritions + 1):
                for pos, entry in enumerate(list(string[150 * i : :])):
                    if entry == " ":
                        string = replace_str_index(string, pos + (150 * i), "\n")
                        break
        return string

    ax.set_title(
        title_reshape(
            re.sub(
                r"\s\s+",
                " ",
                data_set["Diag"].iloc[row_num].replace("ECG anormal", "").replace(";", "\t"),
            )
        ),
        fontsize=20,
        y=0.94,
        backgroundcolor="white",
    )
    # plt.subplots_adjust(top=0.85)

    # add grid
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            "ECG_{}_{}_{}.jpg".format(
                data_set["RestingECG_PatientDemographics_PatientID"].iloc[row_num],
                data_set["RestingECG_TestDemographics_AcquisitionDate"].iloc[row_num],
                data_set["RestingECG_TestDemographics_AcquisitionTime"].iloc[row_num],
            ),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run the script", parents=[get_arguments()])
    args = parser.parse_args()
    dataset = pd.read_csv(args.data_path)
    for i in tqdm(range(dataset.shape[0])) if args.verbose else range(dataset.shape[0]):
        plot_a_lead(data_set=dataset, save_path=args.out_path, row_num=i)
