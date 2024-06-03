# Filename: plot_ecg_from_xml.py

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from CLI_xml2df import tinyxml2df
from PIL import Image


class XMLtoNumpyConverter:
    def __init__(self, xml_path, out_path="tmp/"):
        self.xml_path = xml_path
        self.out_path = out_path

    def load_and_convert(self):
        converter = tinyxml2df(self.xml_path, self.out_path, verbose=True, save=False)
        npy_path = converter.read2flatten()
        return npy_path


def plot_ecg_from_npy(npy_path, title="ECG Plot", out_dir="tmp/", save=False):
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
        newline_index = max_length
        while newline_index < len(input_str) and input_str[newline_index] != " ":
            newline_index += 1
        if newline_index < len(input_str):
            input_str = input_str[:newline_index] + "\n" + input_str[newline_index:]
    return input_str


def plot_xml_files(xml_files, save=False):
    if isinstance(xml_files, str):
        xml_files = [xml_files]  # Convert to list if only one file is provided

    for xml_path in xml_files:
        # print(f"Processing: {xml_path}")
        converter = XMLtoNumpyConverter(xml_path)
        result = converter.load_and_convert()

        plot_ecg_from_npy(npy_path, title=os.path.basename(xml_path), save=save)


def plot_ecg_from_xml(
    xml_path, title="ECG Plot", out_dir="tmp/", save=False, anonymize=True, width=2500
):
    import io
    import os
    import time

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import ticker
    from PIL import Image

    plt.style.use("default")

    converter = tinyxml2df(xml_path, out_dir, verbose=True, save=False)
    npy_path = converter.read2flatten().npy_path[0]

    lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    try:
        lead_data = np.load(npy_path)
        lead_dict = dict(zip(lead_order, np.swapaxes(np.squeeze(lead_data), 0, 1)))
        activation = [0] * 5 + [10] * 50 + [0] * 5

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

        if title:
            ax.set_title(
                insert_newline(title, 150),
                fontsize=30,
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"),
            )

        # Hide x and y axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Hide x and y axis tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", transparent=False)
        img_buffer.seek(0)

        img = Image.open(img_buffer)

        # Rescale image to the specified width
        aspect_ratio = img.height / img.width
        new_height = int(width * aspect_ratio)
        img = img.resize((width, new_height), Image.LANCZOS)

        # Save the rescaled image with the new width
        if save:
            if not anonymize:
                title = os.path.basename(xml_path)
                save_path = os.path.join(out_dir, f"{title}.png")
            else:
                current_time = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(out_dir, f"ECG_{current_time}.png")

            img.save(save_path, dpi=(240, 240))
        else:
            img.show()

        plt.close(fig)
        if os.path.exists(npy_path):
            os.remove(npy_path)
        return {"path": npy_path, "xml_path": xml_path}
    except FileNotFoundError as e:
        print(f"Failed to load npy file at {npy_path}: {e}")
        return {"path": "", "xml_path": xml_path}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"path": "", "xml_path": xml_path}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot ECG from XML files.")
    parser.add_argument("xml_files", nargs="+", help="Path to the XML file or files")
    parser.add_argument(
        "--save", action="store_true", help="Save the plots instead of displaying"
    )
    parser.add_argument("--anonymize", action="store_true", help="Save the plots anonymously")

    args = parser.parse_args()

    for xml_path in args.xml_files:
        plot_ecg_from_xml(xml_path, save=args.save, anonymize=args.anonymize)
