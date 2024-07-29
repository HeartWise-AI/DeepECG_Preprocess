import io
import os
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from CLI_xml2df import tinyxml2df
from PIL import Image


class ECGPlotter(ABC):
    def __init__(
        self,
        dataset="MIMICIV",
        out_dir="tmp/",
        width=2500,
        patient_id_column=None,
        diagnosis_column=None,
    ):
        self.dataset = dataset
        self.out_dir = out_dir
        self.width = width
        self.patient_id_column = patient_id_column
        self.diagnosis_column = diagnosis_column
        self.lead_order, self.amplitude_factor = self._get_dataset_params()

    def _get_dataset_params(self):
        if self.dataset == "MHI":
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
            amplitude_factor = 4.88
        elif self.dataset == "MIMICIV":  # MIMICIV
            lead_order = [
                "I",
                "II",
                "III",
                "aVR",
                "aVF",
                "aVL",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ]
            amplitude_factor = 1.2
        else:
            raise ValueError("Invalid dataset. Choose 'MHI' or 'MIMICIV'.")

        return lead_order, amplitude_factor

    @abstractmethod
    def load_data(self):
        pass

    def plot_ecg(self, title="", save=False, anonymize=True, show_diagnosis=True):
        lead_data = self.load_data()
        if isinstance(self, DataFrameECGPlotter):
            info = self.get_ecg_metadata()
            npy_filename = os.path.basename(info.get("npy_path", ""))
            if info is not None and not anonymize:
                diagnosis_text = (
                    f" - Diagnosis: {info.get('diagnosis', '')}" if show_diagnosis else ""
                )
                title = f"{info.get('patient_id', 'Unknown')} - {info.get('date', 'Unknown Date')} - {info.get('time', 'Unknown Date')}{diagnosis_text}"
                filename = f"{info.get('patient_id', 'Unknown')}_{info.get('date', 'Unknown Date')}_{info.get('time', 'Unknown Date')}_{npy_filename}"
            else:
                diagnosis_text = f"{info.get('diagnosis', '')}" if show_diagnosis else ""
                title = title + diagnosis_text
                filename = f"{npy_filename}"

        if lead_data.shape[0] > 2500:
            lead_data = lead_data[::2, :]  # Take every other sample
        lead_dict = dict(zip(self.lead_order, np.swapaxes(lead_data, 0, 1)))

        fig, ax = self._setup_plot()
        self._plot_leads(ax, lead_dict)
        self._add_lead_labels(ax)

        self._set_title(ax, title)
        plt.tight_layout()
        img, save_path = self._save_or_show_plot(fig, save, anonymize, filename)
        plt.close(fig)
        return save_path

    def _plot_leads(self, ax, lead_dict):
        activation = [0] * 5 + [10] * 50 + [0] * 5

        # Generate the panels
        pannel_1_y = (
            [i + 50 for i in activation]
            + [((i * self.amplitude_factor) / 100) + 50 for i in lead_dict["I"][60:625]]
            + [((i * self.amplitude_factor) / 100) + 50 for i in lead_dict["aVR"][625:1250]]
            + [((i * self.amplitude_factor) / 100) + 50 for i in lead_dict["V1"][1250:1875]]
            + [((i * self.amplitude_factor) / 100) + 50 for i in lead_dict["V4"][1875:2500]]
        )
        pannel_2_y = (
            [i + 15 for i in activation]
            + [((i * self.amplitude_factor) / 100) + 15 for i in lead_dict["II"][60:625]]
            + [((i * self.amplitude_factor) / 100) + 15 for i in lead_dict["aVL"][625:1250]]
            + [((i * self.amplitude_factor) / 100) + 15 for i in lead_dict["V2"][1250:1875]]
            + [((i * self.amplitude_factor) / 100) + 15 for i in lead_dict["V5"][1875:2500]]
        )
        pannel_3_y = (
            [i - 15 for i in activation]
            + [((i * self.amplitude_factor) / 100) - 15 for i in lead_dict["III"][60:625]]
            + [((i * self.amplitude_factor) / 100) - 15 for i in lead_dict["aVF"][625:1250]]
            + [((i * self.amplitude_factor) / 100) - 15 for i in lead_dict["V3"][1250:1875]]
            + [((i * self.amplitude_factor) / 100) - 15 for i in lead_dict["V6"][1875:2500]]
        )
        pannel_4_y = [i - 50 for i in activation] + [
            ((i * self.amplitude_factor) / 100) - 50 for i in lead_dict["II"][60:]
        ]

        ax.axis([0 - 100, 2500 + 100, min(pannel_4_y) - 10, max(pannel_1_y) + 10])
        x = [pos for pos in range(0, len(pannel_1_y))]
        ax.plot(x, pannel_1_y, linewidth=3, color="#000000")
        ax.plot(x, pannel_2_y, linewidth=3, color="#000000")
        ax.plot(x, pannel_3_y, linewidth=3, color="#000000")
        ax.plot(x, pannel_4_y, linewidth=3, color="#000000")

    def _add_lead_labels(self, ax):
        labels = [
            (60, -10, "III"),
            (625, -10, "aVF"),
            (1250, -10, "V3"),
            (1875, -10, "V6"),
            (60, 20, "II"),
            (625, 20, "aVL"),
            (1250, 20, "V2"),
            (1875, 20, "V5"),
            (60, 55, "I"),
            (625, 55, "aVR"),
            (1250, 55, "V1"),
            (1875, 55, "V4"),
            (60, -45, "II"),
        ]
        for x, y, label in labels:
            ax.vlines(x, y - 10, y, linewidth=4, color="black")
            ax.text(x, y, label, fontsize=32, color="black")

    def _setup_plot(self):
        fig, ax = plt.subplots(figsize=(40, 20))
        # Set the background color to white
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.minorticks_on()
        self._setup_grid(ax)
        self._setup_axes(ax)
        return fig, ax

    def _setup_grid(self, ax):
        ax.grid(ls="-", color="red", linewidth=1.2)
        ax.grid(which="minor", ls=":", color="red", linewidth=1)

    def _setup_axes(self, ax):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _set_title(self, ax, title):
        if title:
            ax.set_title(
                self._insert_newline(title, 150),
                fontsize=30,
                color="black",
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"),
            )

    def _save_or_show_plot(self, fig, save, anonymize, title):
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", transparent=False)
        img_buffer.seek(0)
        img = Image.open(img_buffer)

        aspect_ratio = img.height / img.width
        new_height = int(self.width * aspect_ratio)
        img = img.resize((self.width, new_height), Image.LANCZOS)

        if save:
            # Ensure out_dir exists
            os.makedirs(self.out_dir, exist_ok=True)

            if not anonymize:
                save_path = os.path.join(self.out_dir, f"{title}.png")
            else:
                current_time = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(self.out_dir, f"ECG_{current_time}_{title}.png")
            img.save(save_path, dpi=(240, 240))
        else:
            img.show()
        return img, save_path

    @staticmethod
    def _insert_newline(input_str, max_length):
        if len(input_str) > max_length:
            newline_index = max_length
            while newline_index < len(input_str) and input_str[newline_index] != " ":
                newline_index += 1
            if newline_index < len(input_str):
                input_str = input_str[:newline_index] + "\n" + input_str[newline_index:]
        return input_str


class XMLECGPlotter(ECGPlotter):
    def __init__(self, xml_path, **kwargs):
        super().__init__(**kwargs)
        self.xml_path = xml_path

    def load_data(self):
        converter = tinyxml2df(self.xml_path, self.out_dir, verbose=True, save=False)
        npy_path = converter.read2flatten().npy_path[0]
        lead_data = np.squeeze(np.load(npy_path))
        if os.path.exists(npy_path):
            os.remove(npy_path)
        return lead_data


class NPYECGPlotter(ECGPlotter):
    def __init__(self, npy_path, **kwargs):
        super().__init__(**kwargs)
        self.npy_path = npy_path

    def load_data(self):
        return np.squeeze(np.load(self.npy_path))


class DataFrameECGPlotter(ECGPlotter):
    def __init__(
        self,
        parquet_df,
        index,
        npy_path_column="npy_path",
        patient_id_column="new_PatientID",
        diagnosis_column="diagnosis",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.npy_path_column = npy_path_column
        self.patient_id_column = patient_id_column
        self.diagnosis_column = diagnosis_column
        self.df = parquet_df

    def load_data(self):
        npy_path = self.df.iloc[self.index][self.npy_path_column]
        return np.squeeze(np.load(npy_path))

    def get_ecg_metadata(self):
        row = self.df.iloc[self.index]
        info = {}

        try:
            info["patient_id"] = row[self.patient_id_column]
        except KeyError:
            info["patient_id"] = "Unknown"

        try:
            info["date"] = row["RestingECG_TestDemographics_AcquisitionDate"].strftime("%Y-%m-%d")
        except KeyError:
            info["date"] = "Unknown"

        try:
            info["time"] = row["RestingECG_TestDemographics_AcquisitionTime"]
        except KeyError:
            info["time"] = "Unknown"

        try:
            info["diagnosis"] = row[self.diagnosis_column]
        except KeyError:
            info["diagnosis"] = "Unknown"

        try:
            info["npy_path"] = row[self.npy_path_column]
        except KeyError:
            info["npy_path"] = "Unknown"

        return info


def create_ecg_plotter(source_type, source_path, **kwargs):
    if source_type == "xml":
        return XMLECGPlotter(source_path, **kwargs)
    elif source_type == "npy":
        return NPYECGPlotter(source_path, **kwargs)
    elif source_type == "dataframe":
        return DataFrameECGPlotter(source_path, **kwargs)
    else:
        raise ValueError("Invalid source type. Choose 'xml', 'npy', or 'parquet'.")
