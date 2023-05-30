__author__ = "alexis nolin-lapalme"
__email__ = "alexis.nolin-lapalme@umontreal.ca"


import argparse
import base64
import os
import struct
from datetime import datetime

# utils
import numpy as np
import pandas as pd
import xmltodict
from tqdm import tqdm

# click arguments


def get_arguments():
    parser = argparse.ArgumentParser(description="Get argument", add_help=False)

    parser.add_argument(
        "--xml_path",
        metavar="xml_path",
        type=str,
        help="Enter path to xml",
        default="/media/data1/muse_ge/ecg_retrospective",
    )
    parser.add_argument(
        "--out_path", metavar="out_path", type=str, help="Output dir", default="/volume"
    )
    parser.add_argument(
        "--verbose",
        metavar="verbose",
        type=bool,
        help="Do you want a progress bar?",
        default=True,
    )
    parser.add_argument(
        "--save",
        metavar="save",
        type=bool,
        help="Do you want to save [debug option]",
        default=True,
    )
    return parser

class tinyxml2df:
    def __init__(
        self,
        in_path: str,
        out_path: str = "/media/data1/anolin/ECG",
        verbose: bool = True,
        save: bool = True,
    ):
        self.path = in_path
        self.out_path = out_path
        self.verbose = verbose
        self.save = save

    def remove_a_key(self, d, remove_key):
        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == remove_key:
                    del d[key]
                else:
                    self.remove_a_key(d[key], remove_key)

    def decode_ekg_muse(self, raw_wave):
        """
        Ingest the base64 encoded waveforms and transform to numeric
        """
        # covert the waveform from base64 to byte array
        arr = base64.b64decode(bytes(raw_wave, "utf-8"))

        # unpack every 2 bytes, little endian (16 bit encoding)
        unpack_symbols = "".join([char * (len(arr) // 2) for char in "h"])
        byte_array = struct.unpack(unpack_symbols, arr)
        return byte_array

    def decode_ekg_muse_to_array(self, raw_wave, downsample=1):
        """
        Ingest the base64 encoded waveforms and transform to numeric
        downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
        """
        try:
            dwnsmpl = int(1 // downsample)
        except ZeroDivisionError:
            print("You must downsample by more than 0")
        # covert the waveform from base64 to byte array
        arr = base64.b64decode(bytes(raw_wave, "utf-8"))

        # unpack every 2 bytes, little endian (16 bit encoding)
        unpack_symbols = "".join([char * int(len(arr) / 2) for char in "h"])
        byte_array = struct.unpack(unpack_symbols, arr)
        return np.array(byte_array)[::dwnsmpl]

    def xml_to_np_array_file(self, dic, path_to_output=os.getcwd()):
        """
        Upload the ECG as numpy array with shape=[2500,12,1] ([time, leads, 1]).
        The voltage unit should be in 1 mv/unit and the sampling rate should be 250/second (total 10 second).
        The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
        """
        # print(dic)
        try:
            pt_id = dic["RestingECG"]["PatientDemographics"]["PatientID"]
        except:
            print("no PatientID")
            pt_id = "none"
        try:
            AcquisitionDateTime = (
                dic["RestingECG"]["TestDemographics"]["AcquisitionDate"]
                + "_"
                + dic["RestingECG"]["TestDemographics"]["AcquisitionTime"].replace(":", "-")
            )
        except:
            print("no AcquisitionDateTime")
            AcquisitionDateTime = "none"

        # try:
        #     requisition_number = dic['RestingECG']['Order']['RequisitionNumber']
        # except:
        #     print("no requisition_number")
        #     requisition_number = "none"

        # need to instantiate leads in the proper order for the model
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

        """
        Each EKG will have this data structure:
        lead_data = {
            'I': np.array
        }
        """

        lead_data = dict.fromkeys(lead_order)
        # lead_data = {leadid: None for k in lead_order}

        #     for all_lead_data in dic['RestingECG']['Waveform']:
        #         for single_lead_data in lead['LeadData']:
        #             leadname =  single_lead_data['LeadID']
        #             if leadname in (lead_order):
        try:
            for lead in dic["RestingECG"]["Waveform"]:
                for leadid in range(len(lead["LeadData"])):
                    sample_length = len(
                        self.decode_ekg_muse_to_array(lead["LeadData"][leadid]["WaveFormData"])
                    )
                    # sample_length is equivalent to dic['RestingECG']['Waveform']['LeadData']['LeadSampleCountTotal']
                    if sample_length == 5000:
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = self.decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=0.5
                        )
                    elif sample_length == 2500:
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = self.decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=1
                        )
                    else:
                        continue
                # ensures all leads have 2500 samples and also passes over the 3 second waveform

            lead_data["III"] = np.array(lead_data["II"]) - np.array(lead_data["I"])
            lead_data["aVR"] = -(np.array(lead_data["I"]) + np.array(lead_data["II"])) / 2
            lead_data["aVF"] = (np.array(lead_data["II"]) + np.array(lead_data["III"])) / 2
            lead_data["aVL"] = (np.array(lead_data["I"]) - np.array(lead_data["III"])) / 2

            lead_data = {k: lead_data[k] for k in lead_order}
            # drops V3R, V4R, and V7 if it was a 15-lead ECG

            # now construct and reshape the array
            # converting the dictionary to an np.array
            temp = []
            for key, value in lead_data.items():
                temp.append(value)

            # transpose to be [time, leads, ]
            ekg_array = np.array(temp).T

            # expand dims to [time, leads, 1]
            ekg_array = np.expand_dims(ekg_array, axis=-1)

            # Here is a check to make sure all the model inputs are the right shape
            #     assert ekg_array.shape == (2500, 12, 1), "ekg_array is shape {} not (2500, 12, 1)".format(ekg_array.shape )

            # filename = '/ekg_waveform_{}_{}.npy'.format(pt_id, requisition_number)
            filename = f"{pt_id}_{AcquisitionDateTime}.npy"

            path_to_output += filename
            # print(path_to_output)
            with open(path_to_output, "wb") as f:
                np.save(f, ekg_array)
            return path_to_output

        except:
            print("error", dic)
            return None

    def flatten(self, input_node: dict, key_: str = "", output_dict: dict = {}):
        self.remove_a_key(input_node, "Waveform")
        self.remove_a_key(input_node, "OriginalDiagnosis")
        self.remove_a_key(input_node, "Diagnosis")

        if isinstance(input_node, dict):
            for key, val in input_node.items():
                new_key = f"{key_}_{key}" if key_ else f"{key}"
                self.flatten(val, new_key, output_dict)
        elif isinstance(input_node, list):
            for idx, item in enumerate(input_node):
                self.flatten(item, f"{key_}_{idx}", output_dict)
        else:
            output_dict[key_] = input_node

        return output_dict

    def check_abnoramlity(self, data: pd.DataFrame):
        warn = ["Analyse impossible", "ECG anormal"]
        list_abnormality = [0] * data.shape[0]
        for pos, entry in enumerate(data["original_diagnosis"].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1

        for pos, entry in enumerate(data["diagnosis"].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1

        data["warnings"] = list_abnormality
        return data

    def read2flatten(self):
        xml_dict_list = list()
        path_list = list()
        xml_list = list()
        extracted = list()
        npy_list = list()
        dx_txt_list = list()
        original_dx_txt_list = list()

        # print(self.path)
        # files_with_xml = self.path.apply(lambda path: [_ for _ in os.listdir(path) if _.endswith('.xml')]).sum()
        ## Make directory self.out_path if it doesn't exist
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if not os.path.exists(os.path.join(self.out_path, "ecg_npy/")):
            os.makedirs(os.path.join(self.out_path, "ecg_npy/"))
            print("Creating directory")

        # iterate through all the files name verbose or not
        # print("{} | Currently transforming {} xml files from dir {} into dict".format(datetime.now().strftime("%H:%M:%S"),len(files_with_xml),self.path))
        list_files = os.listdir(self.path)
        for file_xml in tqdm(
           list_files, total=len(list_files), desc="Transforming xml files into dict"
        ):
            # with open(os.path.join(self.path,file_xml), 'r') as xml:
            with open(os.path.join(self.path,file_xml)) as xml:
                path_list.append(os.path.join(self.path,file_xml))
                # load
                # *|MARKER_CURSOR|*
                ECG_data_nested = xmltodict.parse(xml.read())
                npy_extracted = self.xml_to_np_array_file(
                    ECG_data_nested, os.path.join(self.out_path, "ecg_npy/")
                )

                try:
                    dx_txt = []
                    for line in ECG_data_nested["RestingECG"]["Diagnosis"]["DiagnosisStatement"]:
                        dx_txt.append(line["StmtText"])
                    ## Flatten array dx_txt and add whitespace between each element
                    dx_txt = " ".join(dx_txt)
                    dx_txt_list.append(dx_txt)
                except:
                    # print(ECG_data_nested)
                    dx_txt_list.append("-1")
                try:
                    original_dx_txt = []
                    for line in ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                        "DiagnosisStatement"
                    ]:
                        original_dx_txt.append(line["StmtText"])
                    original_dx_txt = " ".join(original_dx_txt)
                    original_dx_txt_list.append(original_dx_txt)
                except:
                    original_dx_txt_list.append("-1")

                ECG_data_flatten = self.flatten(ECG_data_nested)

                # append to the list
                ECG_extracted = xml_dict_list.append(ECG_data_flatten.copy())
                if npy_extracted == None:
                    extracted.append("False")
                    npy_list.append("Error")
                else:
                    extracted.append("True")
                    npy_list.append(npy_extracted)

                xml_list.append(os.path.join(self.path,file_xml))

        df = pd.DataFrame(xml_dict_list)
        df["diagnosis"] = dx_txt_list
        df["original_diagnosis"] = original_dx_txt_list
        df["xml_path"] = xml_list
        df["npy_path"] = npy_list
        df["extracted"] = extracted
        df = self.check_abnoramlity(df)

        if self.save == True:
            df.to_csv(
                os.path.join(
                    self.out_path,
                    "df_xml_{}_n_{}.csv".format(datetime.now().strftime("%Y_%m_%d"), df.shape[0]),
                )
            )

        return df


def main(args):
    tinyxml2df(args.xml_path, args.out_path, args.verbose, args.save).read2flatten()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some xml into a df", parents=[get_arguments()]
    )
    args = parser.parse_args()
    main(args)
