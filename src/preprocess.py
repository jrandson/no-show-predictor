import shutil
import os
import re
from glob import glob
import sys
from pathlib import Path
import joblib
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split

from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder

from unidecode import unidecode
import numpy as np
import scipy as sp
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


seed = 142

debug_mode = True

from datetime import timedelta



def format_col(col):
    return "_".join(col.split(' ')).lower()


def replace_rare(df, col, thr=0.0025):
    
    map_values = df[col].value_counts(normalize=True).to_dict()

    def replace(x):
        if map_values[x] < thr:
            return -100
        else:
            return x

    df.loc[:, col] = df[col].apply(lambda x: replace(x))
    
    return df[col]


def categorize_numeric(df_input, feature, n_bins=10):
    
    df = df_input.copy()
    
    df.loc[:, feature] = pd.cut(df[feature], n_bins, precision=0)
    
    return df[feature].apply(lambda x: "%s-%s" %(int(x.left), int(x.right)))


def get_dias_ate_atendimento(df):
    
    df.loc[:, 'data_atendimento'] = df.apply(lambda x:'%s-%s-%s'%(x.ano_atendimento,
                                                      x.mes_atendimento,
                                                      x.dia_mes_atendimento), axis=1)

    df.loc[:, 'data_agendamento'] = df.apply(lambda x:'%s-%s-%s'% (x.ano_agendamento,
                                                            x.mes_agendamento,
                                                            x.dia_mes_agendamento), axis=1)

    df.loc[:, 'data_agendamento'] = pd.to_datetime(df['data_agendamento'], errors='coerce')
    df.loc[:, 'data_atendimento'] = pd.to_datetime(df['data_atendimento'], errors='coerce')
    
    df.loc[:, 'dias_ate_atendimento'] = (df['data_agendamento'] - df['data_atendimento']).dt.days
    
    return df


def remove_outliers(df_input, feature):
    
    df = df_input.copy()
    
    q1, q2, q3 = df[feature].quantile([0.25, 0.5, 0.75])
    IQR = q3 - q1
    
    chart = df[df[feature] < q2 + 1.5 * IQR]
    chart = chart[chart[feature] > q2 - 1.5 * IQR]
    
    return chart

def preprocess(df_input, rm_outlier=False):
    
    df = df_input.copy()
    
    df = df.drop(columns=['Unnamed: 23', 'Legenda'])
    
    columns = {col: format_col(col) for col in df.columns}
    
    df.rename(columns=columns, inplace=True)
    
    df = df.drop_duplicates()
    
    df = df.query('renda_provavel > 0')
    
    df.loc[:, 'renda_provavel'] = df['renda_provavel'] / 1000
    
    df = df.query('tempo_medio_de_agendamento > 0')
    
    df = df.query('idade < 120')
    
    to_reduce = ['microarea', 'codigo_exame', 'bairro', 'cod_convenio']
    
    for col in to_reduce:
        df.loc[:, col] = replace_rare(df, col, thr=0.0025)
        
       
    to_categorize = []
    
    for col in to_categorize:
        df.loc[:, col] = categorize_numeric(df, col).astype(np.object)
    
    df = get_dias_ate_atendimento(df)  
    
    if rm_outlier:
        print("removing outliers")
        
        cols = ['dias_ate_atendimento', 'tempo_medio_de_agendamento', 'renda_provavel', 'idade']        
        for col in cols:
            print(col)
            df = remove_outliers(df, col)
        
        
    columns = [
        'dias_ate_atendimento', 'tempo_medio_de_agendamento', 'renda_provavel', 'idade',        
        'feminino', 'medico_preferencial', 'diretoria', 
        'dia_semana_agendamento',  'dia_semana_atendimento',
        'hora_agendamento', 'hora_atendimento',
        'secao', 'microarea','cod_convenio', 'codigo_exame', 
        'unidade', 'bairro', 
        'no_show']
    
    
    return df[columns]














































