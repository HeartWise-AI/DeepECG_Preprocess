# load packages
import numpy as np
import os
import collections
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from operator import methodcaller
from datetime import datetime
import pandas as pd
from collections.abc import MutableMapping
from lxml import objectify as xml_objectify
from xml.etree import cElementTree as ET

#How to use
# df = BuildDfFromDict('/media/data1/muse_ge/ecg_retrospective').build_the_df()


class BuildDfFromDict():
    """
    directory = "/media/data1/muse_ge/ecg_retrospective"

    returns a df with all the associated dirs of the file and the contents of the original xml
    """
    def __init__(self, directory: str):
        self.dir = directory
    

    def xml_to_dict(self,xml_str):
        """ Convert xml to dict, using lxml v3.4.2 xml processing library """
        def xml_to_dict_recursion(xml_object):
            dict_object = xml_object.__dict__
            if not dict_object:
                return xml_object
            for key, value in dict_object.items():
                dict_object[key] = xml_to_dict_recursion(value)
            return dict_object
        return xml_to_dict_recursion(xml_objectify.fromstring(xml_str))

    def flatten(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def clean(self,list_:str):
        list_.replace('[','')
        list_.replace(']','')
        return list_

    def clean_dict(self,df:pd.DataFrame):
        for col in tqdm(df.columns):
            df[col] = df[col].apply(self.clean)
        return df

    def build_the_df(self, save: bool=True, save_dir: str='/volume', file_name: str="xml_data", loading_bar: bool=True):

        #taking in the flatten dir and using it as a dataframe
        xml_dict_list = list()
        files_with_xml = [_ for _ in os.listdir(self.dir) if _.endswith('.xml')]

        #iterate through all the files name verbose or not
        print("{} | Currently transforming {} xml files from dir {} into dict".format(datetime.now().strftime("%H:%M:%S"),len(files_with_xml),self.dir))
        for file_xml in (tqdm(files_with_xml) if loading_bar else files_with_xml): 
            file_xml = os.path.join(self.dir,file_xml)
            tree = ET.parse(file_xml)
            root = tree.getroot()
            xmlstr = ET.tostring(root, encoding='utf8', method='xml') 
            dict_from_xml = self.xml_to_dict(xmlstr)
            xml_dict_list.append(self.flatten(dict_from_xml))

    
        df_ = self.clean_dict(pd.DataFrame(xml_dict_list))

        if save is True:
            df_.to_csv(os.path.join(save_dir,"{}.csv".format(file_name)))

        return df_
