__author__ = "alexis nolin-lapalme"
__email__ = "alexis.nolin-lapalme@umontreal.ca"


import argparse
import base64
import os
import pprint
import struct
from datetime import datetime

# utils
import numpy as np
import pandas as pd
import xmltodict

pp = pprint.PrettyPrinter(depth=4)
import traceback


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def get_arguments():
    """
    Generate the arguments using CLI
    """
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

        shape_ecg = False

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
                    if sample_length == 5000:  # check if the length is 5000
                        shape_ecg = 5000  # keep in mind the original shape to add it to the output parquet
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = self.decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=0.5
                        )
                    elif sample_length == 2500:  # check if the length is 2500
                        shape_ecg = 2500  # keep in mind the original shape to add it to the output parquet
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = self.decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=1
                        )
                    else:
                        continue
                # ensures all leads have 2500 samples and also passes over the 3 second waveform

            # convert the leads that are measured into the calculated leads
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
            return path_to_output, "empty", shape_ecg

        except:
            print("error", dic)
            return None, "reading_numpy_error", "unknown"

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

    def check_warning_labels(self, data: pd.DataFrame):
        # check if the ECG was validated by MD i.e. if not  'NON VALIDÉ' or 'VALIDÉ PAR MD'
        data["validated by MD"] = np.where(
            (data["RestingECG_TestDemographics_OverreaderLastName"] == "NON VALIDÉ")
            | (data["RestingECG_TestDemographics_OverreaderLastName"] == "VALIDÉ PAR MD"),
            0,
            1,
        )

        # check if lead inversion warning
        substrings_leads = [
            "** Positionnement dérivations non standard, interprétation ECG non disponible **",
            "Inversion electrodes aVR-aVL",
            "Inversion probable des électrodes des membres supérieures; ECG interprété sans en tenir compte",
        ]
        data["diagnosis Inversion electrodes"] = [
            1 if any(substring in i for substring in substrings_leads) else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis Inversion electrodes"] = [
            1 if any(substring in i for substring in substrings_leads) else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if sex-specific analysis was performed
        data["diagnosis sex-specific analysis"] = [
            1 if "ANALYSE ECG SPÉCIFIQUE DE L'ÂGE ET DU SEXE" in i else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis sex-specific analysis"] = [
            1 if "ANALYSE ECG SPÉCIFIQUE DE L'ÂGE ET DU SEXE" in i else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if ECG is from a test demo license
        data["diagnosis demo license"] = [
            1 if "DEMO LICENSE - NOT VALID FOR PRODUCTION USE" in i else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis demo license"] = [
            1 if "DEMO LICENSE - NOT VALID FOR PRODUCTION USE" in i else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if pediatric-based analysis was needed
        data["diagnosis pediatric analysis"] = [
            1 if "ANALYSE PÉDIATRIQUE DE L'ECG" in i else 0 for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis pediatric analysis"] = [
            1 if "ANALYSE PÉDIATRIQUE DE L'ECG" in i else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if low voltage
        substrings_voltage = [
            "BAS VOLTAGE DES QRS",
            "Bas voltage des QRS",
            "bas voltage des QRS",
            "bas voltage",
        ]
        data["diagnosis low voltage"] = [
            1 if any(substring in i for substring in substrings_voltage) else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis low voltage"] = [
            1 if any(substring in i for substring in substrings_voltage) else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if to few or no QRS present
        substrings_QRS = [
            "Analyse impossible; moins de 4 QRS détectés",
            "Analyse impossible; aucun QRS décelable",
        ]
        data["diagnosis low/no QRS available"] = [
            1 if any(substring in i for substring in substrings_QRS) else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis low/no QRS available"] = [
            1 if any(substring in i for substring in substrings_QRS) else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if to few or no QRS present
        data["diagnosis defectuous machine"] = [
            1 if "ATTENTION! matériel d'acquisition peut-être défectueux" in i else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis defectuous machine"] = [
            1 if "ATTENTION! matériel d'acquisition peut-être défectueux" in i else 0
            for i in data["original_diagnosis"].tolist()
        ]

        # check if to few or no QRS present
        data["diagnosis bad ECG"] = [
            1 if "ATTENTION! mauvaise qualité de l’ECG" in i else 0
            for i in data["diagnosis"].tolist()
        ]
        data["original_diagnosis bad ECG"] = [
            1 if "ATTENTION! mauvaise qualité de l’ECG" in i else 0
            for i in data["original_diagnosis"].tolist()
        ]

        return data

    def check_abnoramlity(self, data: pd.DataFrame):
        warn = ["Analyse impossible", "ECG anormal"]
        list_abnormality = [0] * data.shape[0]
        for pos, entry in enumerate(data["original_diagnosis"].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1

        for pos, entry in enumerate(data["diagnosis"].values):
            if any(x in entry for x in warn):
                list_abnormality[pos] = -1

        data["numpy warnings"] = list_abnormality
        return data

    def read2flatten(self):
        xml_dict_list = list()
        path_list = list()
        xml_list = list()
        extracted = list()
        npy_list = list()
        dx_txt_list = list()
        original_dx_txt_list = list()

        reading_xml_error = list()
        original_shape = list()

        DIAG_ERR_LIST = list()
        ORI_DIAG_ERR_LIST = list()

        ## Make directory self.out_path if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(os.path.join(self.out_path, "ecg_npy"), exist_ok=True)

        # iterate through all
        list_files = os.listdir(self.path)
        list_files = [i for i in list_files if "xml" in i]  # select the xml files
        list_files.sort()  # sort, this useful if we want to be reproducible for a minimal added compute time
        list_files.remove("MUSE_20221214_115257_01000.xml")  # this xml exists but empty

        # list_files = list_files[1000000:1200000]
        for file_xml in tqdm(
            list_files, total=len(list_files), desc="Transforming xml files into dict"
        ):
            # with open(os.path.join(self.path,file_xml), 'r') as xml:
            with open(os.path.join(self.path, file_xml)) as xml:
                DIAG_ERR_TYPE = "empty"  # placeholder of diagnosis column error
                ORI_DIAG_ERR_TYPE = "empty"  # placeholder of original diagnosis column error

                BAD_DIAG = False  # bool to check if issue in diag columns
                BAD_ORIG_DIAG = False  # bool to check if issue in original diag columns

                DIAG_FILE_TYPE = "empty"  # placeholder for the type in which the diag is encoded
                ORI_DIAG_FILE_TYPE = (
                    "empty"  # placeholder for the type in which the original diag is encoded
                )

                path_list.append(os.path.join(self.path, file_xml))
                # load
                # *|MARKER_CURSOR|*
                try:
                    ECG_data_nested = xmltodict.parse(xml.read())  # try to open the xml
                    xml_list.append(
                        os.path.join(self.path, file_xml)
                    )  # keep the original xml path

                except:
                    print(
                        f'Can"t read XML: {file_xml}'
                    )  # for error verification print the xml name and continue
                    continue

                # check if there are annotation missing to explain part of the -1
                # ---------------------------------------------------------------

                # Look for empty diag information not format issues

                # Check if the Diagnosis information is missing and why,
                #  - Either the field is empty
                #  - The field contains an empty dictionnary
                if (
                    "Diagnosis" not in ECG_data_nested["RestingECG"]
                    or ECG_data_nested["RestingECG"]["Diagnosis"] == {"Modality": "RESTING"}
                    or ECG_data_nested["RestingECG"]["Diagnosis"] == [{"Modality": "RESTING"}]
                    or ECG_data_nested["RestingECG"]["Diagnosis"]
                    == {"DiagnosisStatement": [{...}, {...}], "Modality": "RESTING"}
                ):
                    BAD_DIAG = True  # if one of these conditions are met put BAD_DIAG to True

                    # One of the potential reasons is the fact the XML is a test
                    if (
                        "medisol"
                        in ECG_data_nested["RestingECG"]["PatientDemographics"][
                            "PatientLastName"
                        ].lower()
                        or "test"
                        in ECG_data_nested["RestingECG"]["PatientDemographics"][
                            "PatientLastName"
                        ].lower()
                    ):  # medisol or medisolution | test
                        DIAG_ERR_TYPE = "This appears to be a test with xml format and patient name"  # the ECG was a test file by the company or someone else

                    # Another reason is a unvalidated ECG
                    elif ECG_data_nested["RestingECG"]["TestDemographics"][
                        "OverreaderLastName"
                    ] in ["VALIDÉ PAR MD", "NON VALIDÉ"]:
                        DIAG_ERR_TYPE = "This appears to be a real ECG but that wasn't validated"

                    else:  # Placeholder for other reasons
                        DIAG_ERR_TYPE = "This appears to be a real ECG but that was seen by an MD without a diag"

                # Check if the Original Diagnosis information is missing and why
                if (
                    "OriginalDiagnosis" not in ECG_data_nested["RestingECG"]
                    or ECG_data_nested["RestingECG"]["OriginalDiagnosis"]
                    == {"Modality": "RESTING"}
                    or ECG_data_nested["RestingECG"]["OriginalDiagnosis"]
                    == [{"Modality": "RESTING"}]
                    or ECG_data_nested["RestingECG"]["OriginalDiagnosis"]
                    == {"DiagnosisStatement": [{...}, {...}], "Modality": "RESTING"}
                ):
                    BAD_ORIG_DIAG = True

                    if (
                        "medisol"
                        in ECG_data_nested["RestingECG"]["PatientDemographics"][
                            "PatientLastName"
                        ].lower()
                        or "test"
                        in ECG_data_nested["RestingECG"]["PatientDemographics"][
                            "PatientLastName"
                        ].lower()
                    ):  # medisol or medisolution | test
                        ORI_DIAG_ERR_TYPE = (
                            "This appears to be a test with xml format and patient name"
                        )

                    elif ECG_data_nested["RestingECG"]["TestDemographics"][
                        "OverreaderLastName"
                    ] in ["VALIDÉ PAR MD", "NON VALIDÉ"]:
                        ORI_DIAG_ERR_TYPE = (
                            "This appears to be a real ECG but that wasn't validated"
                        )
                    else:
                        ORI_DIAG_ERR_TYPE = "This appears to be a real ECG but that was seen by an MD without a diag"

                else:
                    pass  # the ECG looks good

                # Look for format variation
                try:  # assign the file type to allow the correct parsing
                    if BAD_DIAG == False:
                        if isinstance(
                            ECG_data_nested["RestingECG"]["Diagnosis"]["DiagnosisStatement"], list
                        ):
                            DIAG_FILE_TYPE = "list"

                        if isinstance(
                            ECG_data_nested["RestingECG"]["Diagnosis"]["DiagnosisStatement"], dict
                        ):
                            DIAG_FILE_TYPE = "dict"

                except Exception:  # error loop, shouldn;t be activated if everything goes well
                    traceback.print_exc()
                    print(BAD_DIAG)
                    print(DIAG_FILE_TYPE)
                    print(DIAG_ERR_TYPE)
                    print(type(ECG_data_nested["RestingECG"]))
                    pp.pprint(ECG_data_nested["RestingECG"])
                    dx_txt_list.append("somethign seriously wrong here")

                try:  # same thing for original diag
                    if BAD_ORIG_DIAG == False:
                        if isinstance(
                            ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                                "DiagnosisStatement"
                            ],
                            list,
                        ):
                            ORI_DIAG_FILE_TYPE = "list"

                        if isinstance(
                            ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                                "DiagnosisStatement"
                            ],
                            dict,
                        ):
                            ORI_DIAG_FILE_TYPE = "dict"
                except Exception:
                    traceback.print_exc()
                    print(BAD_ORIG_DIAG)
                    print(ORI_DIAG_FILE_TYPE)
                    print(ORI_DIAG_ERR_TYPE)
                    print(type(ECG_data_nested["RestingECG"]))
                    pp.pprint(ECG_data_nested["RestingECG"])
                    original_dx_txt_list.append("somethign seriously wrong here")

                # returns if the npy was extracted (npy_extracted), if there were an error during reading of the leads: reading_xml_error and
                # what was the original length of the ECG i.e. 2500 (250hz) or 5000 (500hz)
                npy_extracted, reading_xml_error, original_shape = self.xml_to_np_array_file(
                    ECG_data_nested, os.path.join(self.out_path, "ecg_npy/")
                )

                # Generate the dianosis and original diagnosis paragraph

                # for diagnosis
                try:  # Flatten array dx_txt and add whitespace between each element
                    dx_txt = []

                    if DIAG_FILE_TYPE == "list":  # if the format is a list list
                        for line in ECG_data_nested["RestingECG"]["Diagnosis"][
                            "DiagnosisStatement"
                        ]:
                            if (
                                line["StmtText"] == None
                            ):  # some although rare diagnosis have a None but information later on
                                continue
                            if (
                                "test" in line["StmtText"].lower() and DIAG_ERR_TYPE == "empty"
                            ):  # although rare the diagnosis is TEST
                                DIAG_ERR_TYPE = "Testing as diagnosis"

                            dx_txt.append(
                                line["StmtText"]
                            )  # if no error add the text to the diagnostic paragraph

                    elif DIAG_FILE_TYPE == "dict":  #  if the format is a list list
                        for k, v in ECG_data_nested["RestingECG"]["Diagnosis"][
                            "DiagnosisStatement"
                        ].items():
                            if k == "StmtText":
                                if (
                                    v == None
                                ):  # some although rare diagnosis have a None but information later on
                                    continue
                                if "test" in v.lower() and DIAG_ERR_TYPE == "empty":
                                    DIAG_ERR_TYPE = "Testing as diagnosis"

                                dx_txt.append(v)

                    else:
                        dx_txt.append("bad diag should be associated with DIAG_ERR_TYPE")

                    dx_txt = " ".join(dx_txt).replace("\n", "")
                    if (
                        "non sauvegard" in dx_txt.lower() and DIAG_ERR_TYPE == "empty"
                    ):  # examen or ecg non sauvegarde
                        DIAG_ERR_TYPE = "ecg not saved"
                    if (
                        "non disponible" in dx_txt.lower() and DIAG_ERR_TYPE == "empty"
                    ):  # ecg or examen non disponible
                        DIAG_ERR_TYPE = "unavailable ecg"
                    dx_txt_list.append(dx_txt)

                except Exception:  # detailed error message for debugging
                    traceback.print_exc()
                    # print(ECG_data_nested)
                    pp.pprint(ECG_data_nested)
                    print("----")
                    print(dx_txt)
                    print(BAD_DIAG)
                    print(DIAG_FILE_TYPE)
                    print(DIAG_ERR_TYPE)
                    print(type(ECG_data_nested["RestingECG"]["Diagnosis"]["DiagnosisStatement"]))
                    pp.pprint(ECG_data_nested["RestingECG"]["Diagnosis"]["DiagnosisStatement"])
                    dx_txt_list.append("somethign seriously wrong here")

                try:
                    original_dx_txt = []
                    if ORI_DIAG_FILE_TYPE == "list":  # is list
                        for line in ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                            "DiagnosisStatement"
                        ]:
                            if (
                                line["StmtText"] == None
                            ):  # some although rare diagnosis have a None but information later on
                                continue
                            if (
                                "test" in line["StmtText"].lower()
                                and ORI_DIAG_ERR_TYPE == "empty"
                            ):
                                ORI_DIAG_ERR_TYPE = "Testing as original diagnosis"

                            original_dx_txt.append(line["StmtText"])

                    elif ORI_DIAG_FILE_TYPE == "dict":  # dict
                        for k, v in ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                            "DiagnosisStatement"
                        ].items():
                            if k == "StmtText":
                                if (
                                    v == None
                                ):  # some although rare diagnosis have a None but information later on
                                    continue
                                if "test" in v.lower() and ORI_DIAG_ERR_TYPE == "empty":
                                    ORI_DIAG_ERR_TYPE = "Testing as original diagnosis"

                                original_dx_txt.append(v)

                    else:
                        original_dx_txt.append(
                            "bad diag should be associated with ORI_DIAG_ERR_TYPE"
                        )

                    original_dx_txt = " ".join(original_dx_txt).replace("\n", "")
                    if (
                        "non sauvegard" in original_dx_txt.lower()
                        and ORI_DIAG_ERR_TYPE == "empty"
                    ):
                        ORI_DIAG_ERR_TYPE = "ecg not saved"
                    if (
                        "non disponible" in original_dx_txt.lower()
                        and ORI_DIAG_ERR_TYPE == "empty"
                    ):
                        ORI_DIAG_ERR_TYPE = "unavailable ecg"
                    original_dx_txt_list.append(original_dx_txt)

                except Exception:
                    traceback.print_exc()
                    # print(ECG_data_nested)
                    pp.pprint(ECG_data_nested)
                    print("----")
                    print(original_dx_txt)
                    print(BAD_ORIG_DIAG)
                    print(ORI_DIAG_FILE_TYPE)
                    print(ORI_DIAG_ERR_TYPE)
                    print(
                        type(
                            ECG_data_nested["RestingECG"]["OriginalDiagnosis"][
                                "DiagnosisStatement"
                            ]
                        )
                    )
                    pp.pprint(
                        ECG_data_nested["RestingECG"]["OriginalDiagnosis"]["DiagnosisStatement"]
                    )
                    original_dx_txt_list.append("somethign seriously wrong here")

                DIAG_ERR_LIST.append(DIAG_ERR_TYPE)  # add the error types
                ORI_DIAG_ERR_LIST.append(ORI_DIAG_ERR_TYPE)

                # append to the list
                ECG_data_flatten = self.flatten(ECG_data_nested)
                ECG_extracted = xml_dict_list.append(ECG_data_flatten.copy())
                if npy_extracted == None:
                    extracted.append("False")
                    npy_list.append("Error")
                else:
                    extracted.append("True")
                    npy_list.append(npy_extracted)

        # generate the final df and add the associated columns
        df = pd.DataFrame(xml_dict_list)
        df["diagnosis"] = dx_txt_list
        df["original_diagnosis"] = original_dx_txt_list
        df["xml_path"] = xml_list
        df["npy_path"] = npy_list
        df["extracted"] = extracted
        df["error_reading_diag"] = DIAG_ERR_LIST
        df["error_reading_original_diag"] = ORI_DIAG_ERR_LIST
        df["reading_xml_error"] = reading_xml_error
        df["original_shape"] = original_shape

        df = self.check_abnoramlity(df)
        df = self.check_warning_labels(df)

        if self.save == True:
            df.to_parquet(
                os.path.join(
                    self.out_path,
                    "df_xml_{}_n_{}_FINAL.parquet".format(
                        datetime.now().strftime("%Y_%m_%d"), df.shape[0]
                    ),
                )
            )

        return df


def main(xml_path, out_path, verbose, save):
    tinyxml2df(xml_path, out_path, verbose, save).read2flatten()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some xml into a df", parents=[get_arguments()]
    )
    args = parser.parse_args()
    main(args.xml_path, args.out_path, args.verbose, args.save)
