import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def insert_newline(input_str, max_length):
    """
    Inserts a newline character at the specified maximum length in the input string.

    Parameters:
    - input_str (str): The input string.
    - max_length (int): The maximum number of characters before inserting a newline.

    Returns:
    - str: The modified string with newline characters.
    """
    if len(input_str) > max_length:
        # Find the index to insert the newline character
        newline_index = max_length
        while newline_index < len(input_str) and input_str[newline_index] != " ":
            newline_index += 1

        # Insert the newline character
        if newline_index < len(input_str):
            input_str = input_str[:newline_index] + "\n" + input_str[newline_index:]

    return input_str


def plot_from_parquet(
    parquet,
    patient_id=None,
    date=None,
    time=None,
    plot_original_diagnosis=True,
    index=None,
    save=False,
    anonym=True,
    out_dir="/volume",
):
    """
    This function plots an EKF for a patient given a parquet it has two main setups

    1. index selection
        set an index in the parquet to plot, make sure to leave atient_id=None, date=None, time=None
    2. selection by patient
        set a patient id, date and time and will plot that ecg, let the index variable to none

    Parameters:
    - parquet: parquet file
    - patient_id: patient ID to plot
    - date: date of the ECG
    - time: time at which the ECG was acquired
    - plot_original_diagnosis: plot the original diagnosis, if set to False will plot the diagnosis columns
    - index: index in the parquet to plot
    - save: to save the PNG
    - anonym: to save the PNG anonymously
    - out_dir: save directory

    Returns:
    - a png or shows the plot
    """

    # assert isinstance(parquet, str),    'please input a parquet string'

    if index == None:
        assert any(
            v is not None for v in [patient_id, date, time]
        ), "please select either a search by setting: index, or by chosing: patient_id, data, time"

    if index != None:
        assert all(
            v is None for v in [patient_id, date, time]
        ), "Seems you select search by index, please make sure patient_id, data, time are not set"

    # find the row
    if index != None:
        index = int(index)
        line = parquet.iloc[index]
        npy_path = line["npy_path"]

        if plot_original_diagnosis == True:
            title = line["original_diagnosis"]
        else:
            title = line["diagnosis"]

        patient_id = line["RestingECG_PatientDemographics_PatientID"]
        date = line["RestingECG_TestDemographics_AcquisitionDate"]
        time = line["RestingECG_TestDemographics_AcquisitionTime"]

    else:
        assert (
            patient_id in parquet.RestingECG_PatientDemographics_PatientID.tolist()
        ), "the patient ID doesn't exits"
        assert (
            date in parquet.RestingECG_TestDemographics_AcquisitionDate.tolist()
        ), "the date doesn't seem to exist check you wrote it in format month-day-year"
        assert (
            time in parquet.RestingECG_TestDemographics_AcquisitionTime.tolist()
        ), "the time doesn't seem to exist"

        line = parquet[
            (parquet.RestingECG_PatientDemographics_PatientID == patient_id)
            & (parquet.RestingECG_TestDemographics_AcquisitionDate == date)
            & (parquet.RestingECG_TestDemographics_AcquisitionTime == time)
        ]

        if len(line) > 1:
            print("WARNING: more that one match: taking the first one")
            line = line.iloc[0]
            npy_path = line["npy_path"]
            print(npy_path)

            if plot_original_diagnosis == True:
                title = line["original_diagnosis"]
            else:
                title = line["diagnosis"].tolist()

        else:
            npy_path = line["npy_path"].tolist()[0]
            if plot_original_diagnosis == True:
                title = line["original_diagnosis"].tolist()[0]
            else:
                title = line["diagnosis"].tolist()[0]

    print(f"Plotting ID {patient_id} at {date} {time}")

    lead_order = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    lead_dict = dict(zip(lead_order, np.swapaxes(np.squeeze(np.load(npy_path)), 0, 1)))
    activation = [0] * 5 + [10] * 50 + [0] * 5
    # generate the pannels
    pannel_1_y = (
        [i + 50 for i in activation]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["I"][60:625]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["aVR"][625:1250]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["V1"][1250:1875]]
        + [((i * 4.88) / 100) + 50 for i in lead_dict["V4"][1875:2500]]
    )
    pannel_2_y = (
        [i + 15 for i in activation]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["II"][60:625]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["aVL"][625:1250]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["V2"][1250:1875]]
        + [((i * 4.88) / 100) + 15 for i in lead_dict["V5"][1875:2500]]
    )
    pannel_3_y = (
        [i - 15 for i in activation]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["III"][60:625]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["aVF"][625:1250]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["V3"][1250:1875]]
        + [((i * 4.88) / 100) - 15 for i in lead_dict["V6"][1875:2500]]
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

    x = [pos for pos in range(0, len(pannel_1_y))]
    ax.plot(x, pannel_1_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_2_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_3_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_4_y, linewidth=3, color="#000000")

    ax.set_title(
        insert_newline(title, 150),
        fontsize=30,
        bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"),
    )

    # add grid
    plt.tight_layout()

    if save == True:
        if anonym == False:
            plt.savefig(os.path.join(out_dir, f"{patient_id}_{date}_{time}.png"))
        else:
            plt.savefig(os.path.join(out_dir, "ECG.png"))

    else:
        plt.show()
