import os, pathlib, sys
import yaml
import pandas as pd 
from _detect_extension import detect_delimiter
class DataReader(object):
    file_name: str
    idx: str
    decimal: str
    delimiter: str

    def __init__(self,
                 path):
        self.path = path

    def read_csv(self, file_name,
                 idx='Date',
                 decimal='.',
                 delimiter=','):

        self.file_name = file_name
        self.idx = idx
        self.decimal = decimal

        with open(os.path.join(self.path, self.file_name)) as myfile:
            firstline = myfile.readline()
        myfile.close()
        delimiter = detect_delimiter(firstline)

        self.delimiter = delimiter
        if os.path.exists(os.path.join(self.path, self.file_name)):
            data = pd.read_csv(os.path.join(self.path, self.file_name),
                               decimal=self.decimal, sep=self.delimiter)
            data[self.idx] = pd.to_datetime(data[self.idx])
            data.set_index(self.idx, inplace=True)
            # making sure it's on chronological order
            data.sort_index(inplace=True)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data

    def read_excel(self, file_name,
                   sheet_name,
                   idx='Date'):

        self.file_name = file_name
        self.idx = idx
        self.sheet_name = sheet_name
        engine = 'openpyxl' if file_name.split('.')[-1]=='xlsx' else 'xlrd'
        if os.path.exists(os.path.join(self.path, self.file_name)):
            data = pd.read_excel(os.path.join(self.path, self.file_name), sheet_name=self.sheet_name, engine=engine)
            data[self.idx] = pd.to_datetime(data[self.idx])
            data.set_index(self.idx, inplace=True)
            # making sure it's on chronological order
            data.sort_index(inplace=True)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data

    def read_simple_csv(self, file_name, decimal='.', delimiter=','):

        self.file_name = file_name
        self.decimal = decimal

        with open(os.path.join(self.path, self.file_name)) as myfile:
            firstline = myfile.readline()
        myfile.close()
        delimiter = detect_delimiter(firstline)
        self.delimiter = delimiter
        if os.path.exists(os.path.join(self.path, self.file_name)):
            data = pd.read_csv(os.path.join(self.path, self.file_name),
                           decimal=self.decimal, sep=self.delimiter)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data

    def read_simple_excel(self, file_name, sheet_name):

        self.file_name = file_name
        self.sheet_name = sheet_name
        engine = 'openpyxl' if file_name.split('.')[-1]=='xlsx' else 'xlrd'
        if os.path.exists(os.path.join(self.path, self.file_name)):
            data = pd.read_excel(os.path.join(self.path, self.file_name), sheet_name=self.sheet_name, engine=engine)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data

    def read_yml(self, file_name):

        self.file_name = file_name
        if os.path.exists(os.path.join(self.path, self.file_name)):
            with open(os.path.join(self.path, self.file_name)) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data

    def read_json(self, file_name, orient=None):

        self.file_name = file_name
        file_path = os.path.join(self.path, self.file_name)
        if os.path.exists(file_path):
            data = pd.read_json(file_path, orient=orient)
            # making sure it's on chronological order
            data.sort_index(inplace=True)
        else:
            data = None
            raise Exception(f"ERROR: File not found. {self.path}{self.file_name}")
        return data
