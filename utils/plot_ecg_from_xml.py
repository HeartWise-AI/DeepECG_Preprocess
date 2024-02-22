# Filename: plot_ecg_from_xml.py

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from CLI_xml2df import tinyxml2df


class XMLtoNumpyConverter:
    def __init__(self, xml_path, out_path="/tmp"):
        self.xml_path = xml_path
        self.out_path = out_path

    def load_and_convert(self):
        converter = tinyxml2df(self.xml_path, self.out_path, verbose=True, save=False)
        npy_path = converter.read2flatten()
        return npy_path


def plot_ecg_from_npy(npy_path, title="ECG Plot", out_dir="/tmp", save=False):
    ekg_array = np.load(npy_path)
    lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lead_dict = dict(zip(lead_order, np.swapaxes(np.squeeze(ekg_array), 0, 1)))

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, lead in enumerate(lead_order):
        ax.plot(lead_dict[lead] + i * 1000, label=lead)  # Offset each lead for clarity

    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, f"{title}.png"))
    else:
        plt.show()


def plot_xml_files(xml_files, save=False):
    if isinstance(xml_files, str):
        xml_files = [xml_files]  # Convert to list if only one file is provided

    for xml_path in xml_files:
        print(f"Processing: {xml_path}")
        converter = XMLtoNumpyConverter(xml_path)
        result = converter.load_and_convert()
        npy_path = result.npy_path[
            0
        ]  # Assuming this is the correct way to access the waveform array
        plot_ecg_from_npy(npy_path, title=os.path.basename(xml_path), save=save)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot ECG from XML files.")
    parser.add_argument("xml_files", nargs="+", help="Path to the XML file or files")
    parser.add_argument(
        "--save", action="store_true", help="Save the plots instead of displaying"
    )

    args = parser.parse_args()

    plot_xml_files(args.xml_files, save=args.save)
