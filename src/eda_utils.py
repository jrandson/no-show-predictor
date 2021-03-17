
import os
from glob import glob
import sys
import math
import logging
from pathlib import Path
from unidecode import unidecode
import re
from datetime import datetime

import numpy as np
import scipy as sp
import sklearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def categorize_numeric(df_input, feature, n_bins = 10):
    
    df = df_input.copy()
    
    df[feature] = pd.cut(df[feature], n_bins, precision=0)
    return df[feature].apply(lambda x: "%s-%s" %(int(x.left), int(x.right)))

    
def insert_note(note):
    with open("notes.txt", mode='a+') as f:
        date = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        note = """\n{} : {} \n---------------""".format(date, note)
        f.write(note)
        
def load_notes():
    with open("notes.txt", mode='r') as f:
        print(f.read())
