# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:29:20 2022

@author: Luis
"""

import pandas as pd
import numpy as np
import sidrapy
from pprint import pprint 
import matplotlib.pyplot as plt

link = "https://apisidra.ibge.gov.br/values/t/8159/n1/all/v/11599,11600/p/all/c544/129314/d/v11599%205,v11600%205"

raiz = "https://apisidra.ibge.gov.br/values/t/"

#t (table_code) =8159
#n (territorial_level) = 1
#n/ (ibge_territorial_code) = all
#v (variable) = 11599,11600
#p (period) = all
#c/ (classification) = 544/129314

pimpf = sidrapy.get_table(table_code="8159", territorial_level="1", ibge_territorial_code="all", variable="11599,11600", classification="544/129314", period="all")

type(pimpf)
pimpf.head()
pimpf.tail()
pimpf.columns = pimpf.iloc[0]
pimpf = pimpf.iloc[1:, :]
pimpf['Valor'] = pimpf['Valor'].astype(float)

pim_nsa = pimpf.loc[pimpf['Variável (Código)']=="11599"]
pim_sa = pimpf.loc[pimpf['Variável (Código)']=="11600"]

pim_nsa = pim_nsa.rename(columns = {"Valor" : "pim_nsa", "Mês (Código)" : "date"})[['pim_nsa', 'date']]
pim_sa = pim_sa.rename(columns = {"Valor" : "pim_sa", "Mês (Código)" : "date"})[['pim_sa', 'date']]

pim_nsa.index = pd.to_datetime(pim_nsa['date'], format = "%Y%m")
pim_sa.index = pd.to_datetime(pim_sa['date'], format = "%Y%m")

pim_nsa = pim_nsa.drop(columns = ['date'])
pim_sa = pim_sa.drop(columns = ['date'])


#plot
pim = pd.merge(pim_nsa, pim_sa, left_index = True, right_index = True)
pim = pim.reset_index()

plt.plot(pim['date'], pim['pim_nsa'], color='red')
plt.plot(pim['date'], pim['pim_sa'], color='blue')
plt.title('Industrial Production', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Index', fontsize=14)
plt.legend(['NSA','SA'])
plt.grid(True)
plt.show()

#Seas
import statsmodels
import os

genhol = pd.read_csv('C:\ECONOMIA_20200314\SEAS\Regressores\Regs_R.csv', delimiter=";") 

genhol['date'] = pd.date_range('2001-01-01',periods=len(genhol['carnaval']),
                               freq='MS') #.strftime("%Y-%b").tolist()

# genhol['outliers'] = np.where((genhol['date']=="2008-11-01") |
#                               (genhol['date']=="2008-12-01") |
#                               (genhol['date']=="2018-04-01") |
#                               (genhol['date']=="2020-03-01") |
#                               (genhol['date']=="2020-04-01")
#                               , 1, 0)

genhol.index = pd.to_datetime(genhol['date'], format = "%Y%m")

genhol = genhol.drop(columns = ['Unnamed: 0','date','"aux"'])

XPATH = os.chdir("C:/ECONOMIA_20200314/SEAS/x13as")

pim_nsa_genhol = statsmodels.tsa.x13.x13_arima_analysis(pim_nsa, 
                 maxorder=(1, 1), maxdiff=None, diff=(1,1), 
                 exog=genhol, log=False, outlier=True, 
                 trading=True, forecast_periods=None, retspec=True,
                 speconly=False, start=None, freq=None,
                 print_stdout=False, x12path=XPATH, 
                 prefer_x13=True)

print(pim_nsa_genhol.results.split('\x0c')[0])
#print(pim_nsa_genhol.results.split('\x0c')[1])
#print(pim_nsa_genhol.results.split('\x0c')[2])
print(pim_nsa_genhol.results.split('\x0c')[3])
#print(pim_nsa_genhol.results.split('\x0c')[4])

pim_sa_genhol = pd.Series(pim_nsa_genhol.seasadj.values,
                          index=pim_sa.index.values)
pim_sa_genhol = pim_sa_genhol.to_frame()
pim_sa_genhol.columns = ['pim_sa_genhol']


ibge_genhol = pd.merge(pim_sa, pim_sa_genhol, left_index = True, right_index = True)
ibge_genhol = ibge_genhol.reset_index()

plt.plot(ibge_genhol['date'], ibge_genhol['pim_sa'], color='red')
plt.plot(ibge_genhol['date'], ibge_genhol['pim_sa_genhol'], color='blue')
plt.title('Industrial Production', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Index', fontsize=14)
plt.legend(['SA_IBGE','SA_Genhol'])
plt.grid(True)
plt.show()