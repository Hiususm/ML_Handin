# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:38:14 2019

@author: franc
"""

import pandas as pd


# Had to split the Rating download from Wharton in 2 files
# Append Ratings File to oneanother (Output -> df2)
df0 = pd.read_csv(r"SP500_Ratings_1st.csv")
print(df0.shape)
df1 = pd.read_csv(r"SP500_Ratings_2nd.csv")
print(df1.shape)
df2 = df0.append(df1, ignore_index = True)
print(df2.shape)


# Join the new ratings dataframe with the sp500_companylist (Output -> df4)
df3 = pd.read_csv(r"SP500_CompanyList.csv")
df3.rename(columns={'TICKER':'tic'}, inplace=True)
print(df3.shape)
df2.rename(str.lower, axis='columns')
print(df2.shape)

df4 = df2.merge(df3, how = 'left', on='tic')
print(df4.shape)


# Join df4 with the SP500_Ratios file
df4.rename(columns={'datadate':'public_date', 'PERMNO' : 'permno'}, inplace=True)
df5 = pd.read_csv(r"firm_lvl_ratios.csv")
print(df5.shape)
df6 = df5.merge(df4, how='left', on=['permno', 'public_date'])
print(df6.shape)


# Drop rows with splticrm = NAN
df_out = df6.dropna(axis=0, subset=['splticrm'])
print(df_out.shape)
df_out.to_csv("SP500_data.csv", index = False)
