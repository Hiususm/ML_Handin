# In[0] Library Import

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

import timeit


# In[2]: Import data

startzeitDaten = timeit.default_timer()

data = pd.read_csv(r"Data\SP500_data_new.csv"
                   ,parse_dates = ["adate", "qdate", "public_date"], dayfirst = True)#, index_col=["gvkey", "datadate"]) 
data_NaN = data.dropna()
data_y = data.dropna(subset =["splticrm"])

Names1 = pd.read_excel(r"Data\Names1.xlsx", header = 0)
Names1 = Names1.drop(["Data Type", "Help"], axis = 1)
Names1.columns = ["Name", "ExName"]
Names2 = pd.read_excel(r"Data\Names2.xlsx", header = 0)
Names2 = Names2.drop(["Data Type","Help"], axis = 1)
Names2.columns = ["Name", "ExName"]

y1 = data_NaN.splticrm
y2 = data_y.splticrm

y1 = pd.Categorical(y1, ordered = True, categories = ['AAA', 'AA+', 'AA', 'AA-','A+', 'A', 'A-',
                                                      'BBB+', 'BBB', 'BBB-','BB+', 'BB']).codes
y2 = pd.Categorical(y2, ordered = True, categories = ['AAA', 'AA+', 'AA', 'AA-','A+', 'A', 'A-',
                                                      'BBB+', 'BBB', 'BBB-','BB+', 'BB', "BB-", "B+",
                                                      "B", "B-", "CCC+", "CCC", "D"]).codes

X1 = data_NaN.drop(["sic", "naics", "splticrm", "adate", "qdate","gvkey","conm",
                    "cusip", "tic", "CUSIP", "NCUSIP", "NWPERM",
                    "spcindcd", "spcseccd", "tic", "cusip", "public_date",
                    "PERMCO"], axis = 1)
X1_column_names = X1.columns.tolist()

X2 = data_y.drop(["sic", "naics", "splticrm", "adate", "qdate","gvkey","conm",
                    "cusip", "tic", "CUSIP", "NCUSIP", "NWPERM",
                    "spcindcd", "spcseccd", "tic", "cusip", "public_date",
                    "PERMCO"], axis = 1)
X2_column_names = X2.columns.tolist()

endzeitDaten = timeit.default_timer()
print("Zeit bis die Daten eingelesen sind: ",endzeitDaten-startzeitDaten)

# In[] Eit Data
#Here we replace NaNs with the medain in the respective class, where possible else =0
SimImp = SimpleImputer(missing_values = np.nan, strategy = "median")#, copy = False)

X2_0 = SimImp.fit_transform(X2[y2==0])
X2_1 = SimImp.fit_transform(X2[y2==1])
X2_2 = SimImp.fit_transform(X2[y2==2])
X2_3 = SimImp.fit_transform(X2[y2==3])
X2_4 = SimImp.fit_transform(X2[y2==4])
X2_5 = SimImp.fit_transform(X2[y2==5])
X2_6 = SimImp.fit_transform(X2[y2==6])
X2_7 = SimImp.fit_transform(X2[y2==7])
X2_8 = SimImp.fit_transform(X2[y2==8])
X2_9 = SimImp.fit_transform(X2[y2==9])
X2_10 =SimImp.fit_transform(X2[y2==10])
X2_11 =SimImp.fit_transform(X2[y2==11])
X2_12 =SimImp.fit_transform(X2[y2==12])
X2_13 =SimImp.fit_transform(X2[y2==13])
X2_14 =SimImp.fit_transform(X2[y2==14])
X2_15 =(X2[y2==15])
X2_16 =(X2[y2==16])
X2_17 =(X2[y2==17]) 
X2_18 =(X2[y2==18])

#X2_17, X2_18 is left out because it has too few "CCC" (2) and "D" (4) in the data

list_namesX = [X2_1, X2_2, X2_3, X2_4, X2_5, X2_6, X2_7, X2_8, X2_9, X2_10,
               X2_11, X2_12, X2_13, X2_14, X2_15, X2_16, X2_17, X2_18]

X2_imp = pd.DataFrame(X2_0, columns = X2_column_names)

for i in list_namesX :
    i = pd.DataFrame(i, columns = X2_column_names)
    X2_imp = X2_imp.append(i)

X2_imp = np.nan_to_num(X2_imp)
X2_imp = pd.DataFrame(X2_imp, columns = X2_column_names)

corr_frame1 = X1.assign(splticrm = y1)
#corr_frame2 = X2_imp.assign(splticrm = y2)

#Konstruiere naives balanced Datenset Schaut euch mal "imbalanced-learn" an online
rebaStart = timeit.default_timer()

y1_0,X1_0 =resample(y1[y1==0], X1[y1==0], replace = True, n_samples = 55, random_state = 0)
y1_1,X1_1 =resample(y1[y1==1], X1[y1==1], replace = True, n_samples = 55, random_state = 0)
y1_2,X1_2 =resample(y1[y1==2], X1[y1==2], replace = True, n_samples = 55, random_state = 0)
y1_3,X1_3 =resample(y1[y1==3], X1[y1==3], replace = True, n_samples = 55, random_state = 0)
y1_4,X1_4 =resample(y1[y1==4], X1[y1==4], replace = True, n_samples = 55, random_state = 0)
y1_5,X1_5 =resample(y1[y1==5], X1[y1==5], replace = True, n_samples = 55, random_state = 0)
y1_6,X1_6 =resample(y1[y1==6], X1[y1==6], replace = True, n_samples = 55, random_state = 0)
y1_7,X1_7 =resample(y1[y1==7], X1[y1==7], replace = True, n_samples = 55, random_state = 0)
y1_8,X1_8 =resample(y1[y1==8], X1[y1==8], replace = True, n_samples = 55, random_state = 0)
y1_9,X1_9 =resample(y1[y1==9], X1[y1==9], replace = True, n_samples = 55, random_state = 0)
y1_10,X1_10 =resample(y1[y1==10], X1[y1==10], replace = True, n_samples = 55, random_state = 0)
y1_11,X1_11 =resample(y1[y1==11], X1[y1==11], replace = True, n_samples = 55, random_state = 0)

list_namesX1 = [X1_1, X1_2, X1_3, X1_4, X1_5, X1_6, X1_7, X1_8, X1_9, X1_10, X1_11]
list_namesy1 = [y1_1, y1_2, y1_3, y1_4, y1_5, y1_6, y1_7, y1_8, y1_9, y1_10, y1_11]

X1_re = X1_0
for i in list_namesX1:
    X1_re = X1_re.append(i)

y1_re = pd.DataFrame(y1_0, columns = ["splticrm"])
for i in list_namesy1: 
    i = pd.DataFrame(i, columns = ["splticrm"])
    y1_re = y1_re.append(i)

y2_0,X2_0 =resample(y2[y2==0], X2_imp[y2==0], replace = True, n_samples = 55, random_state = 0)
y2_1,X2_1 =resample(y2[y2==1], X2_imp[y2==1], replace = True, n_samples = 55, random_state = 0)
y2_2,X2_2 =resample(y2[y2==2], X2_imp[y2==2], replace = True, n_samples = 55, random_state = 0)
y2_3,X2_3 =resample(y2[y2==3], X2_imp[y2==3], replace = True, n_samples = 55, random_state = 0)
y2_4,X2_4 =resample(y2[y2==4], X2_imp[y2==4], replace = True, n_samples = 55, random_state = 0)
y2_5,X2_5 =resample(y2[y2==5], X2_imp[y2==5], replace = True, n_samples = 55, random_state = 0)
y2_6,X2_6 =resample(y2[y2==6], X2_imp[y2==6], replace = True, n_samples = 55, random_state = 0)
y2_7,X2_7 =resample(y2[y2==7], X2_imp[y2==7], replace = True, n_samples = 55, random_state = 0)
y2_8,X2_8 =resample(y2[y2==8], X2_imp[y2==8], replace = True, n_samples = 55, random_state = 0)
y2_9,X2_9 =resample(y2[y2==9], X2_imp[y2==9], replace = True, n_samples = 55, random_state = 0)
y2_10,X2_10 =resample(y2[y2==10], X2_imp[y2==10], replace = True, n_samples = 55, random_state = 0)
y2_11,X2_11 =resample(y2[y2==11], X2_imp[y2==11], replace = True, n_samples = 55, random_state = 0)
y2_12,X2_12 =resample(y2[y2==12], X2_imp[y2==12], replace = True, n_samples = 55, random_state = 0)
y2_13,X2_13 =resample(y2[y2==13], X2_imp[y2==13], replace = True, n_samples = 55, random_state = 0)
y2_14,X2_14 =resample(y2[y2==14], X2_imp[y2==14], replace = True, n_samples = 55, random_state = 0)
y2_15,X2_15 =resample(y2[y2==15], X2_imp[y2==15], replace = True, n_samples = 55, random_state = 0)
y2_16,X2_16 =resample(y2[y2==16], X2_imp[y2==16], replace = True, n_samples = 55, random_state = 0)
y2_17,X2_17 =resample(y2[y2==17], X2_imp[y2==17], replace = True, n_samples = 55, random_state = 0)
y2_18,X2_18 =resample(y2[y2==18], X2_imp[y2==18], replace = True, n_samples = 55, random_state = 0)

list_namesX = [X2_1, X2_2, X2_3, X2_4, X2_5, X2_6, X2_7, X2_8, X2_9, X2_10,
               X2_11, X2_12, X2_13, X2_14, X2_15, X2_16, X2_17, X2_18]
list_namesy = [y2_1, y2_2, y2_3, y2_4, y2_5, y2_6, y2_7, y2_8, y2_9, y2_10,
               y2_11, y2_12, y2_13, y2_14, y2_15, y2_16, y2_17, y2_18]

X2_re = X2_0
for i in list_namesX:
    X2_re = X2_re.append(i)

y2_re = pd.DataFrame(y2_0, columns = ["splticrm"])
for i in list_namesy: 
    i = pd.DataFrame(i, columns = ["splticrm"])
    y2_re = y2_re.append(i)
    
rebaEnd = timeit.default_timer()
print("Rebalance zeit : ", rebaEnd-rebaStart)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_re, y1_re ,test_size = 0.2, train_size = 300)#, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_re, y2_re, test_size = 0.2, train_size = 300)#, random_state = 0)

sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X2_train = sc.fit_transform(X2_train)

#Number of PCA components
pcaN = 8

PCA1_train = PCA(n_components = pcaN).fit(X1_train).transform(X1_train)
PCA2_train = PCA(n_components = pcaN).fit(X2_train).transform(X2_train)

PCA1 = PCA(n_components = pcaN).fit(X1).transform(X1)
PCA2 = PCA(n_components = pcaN).fit(X2_imp).transform(X2_imp)

PCA_Labels = np.arange(start = 0, stop = pcaN +1, step = 1)

y1_train = np.array(y1_train).ravel()
y2_train = np.array(y2_train).ravel()

forest = RandomForestClassifier(max_depth = None)
forestBoost = GradientBoostingClassifier(max_depth = None)
MLP = MLPClassifier()
svm = SVC()
knn = KNeighborsClassifier()

endzeitDaten = timeit.default_timer()
print("Zeit bis die Daten eingelesen sind: ",endzeitDaten-startzeitDaten)

X2 = X2_imp

# In[2] Random Forest 

""" X1 """

forest.fit(X1 , y1) 
importances = forest.feature_importances_
indices = np.argsort(importances )[::-1] 

A1 = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    A1.iloc[count, 0] = X1_column_names[i]
    A1.iloc[count, 1] = (importances[count])
    count = count +1
    
A1.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("RND-Forest X1")
print(A1)

A1 = A1.iloc[:40] 
A1Plot =sns.barplot(x=A1["Name"], y=A1["Wert"], palette="husl")
for item in A1Plot.get_xticklabels():
    item.set_rotation(90)
plt.show()

""" X2 """

forest.fit(X2, y2)
importances = forest.feature_importances_
indices = np.argsort(importances )[::-1] 

A2 = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    A2.iloc[count, 0] = X1_column_names[i]
    A2.iloc[count, 1] = (importances[count])
    count = count +1
    
A2.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("RND-Forest X2")
print(A2)

A2 = A2.iloc[:40] 
A2Plot =sns.barplot(x=A2["Name"], y=A2["Wert"], palette="husl")
for item in A2Plot.get_xticklabels():
    item.set_rotation(90)
plt.show()

""" PCA1 """

forest.fit(PCA1 , y1) 
importances = forest.feature_importances_
indices = np.argsort(importances )[::-1] 

A1_PCA = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    A1_PCA.iloc[count, 0] = PCA_Labels[i]
    A1_PCA.iloc[count, 1] = (importances[count])
    count = count +1
    
A1_PCA.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("RND-Forest PCA1")
print(A1_PCA)

A1_PCA = A1_PCA.iloc[:40] 
A1_PCAPlot =sns.barplot(x=A1_PCA["Name"], y=A1_PCA["Wert"], palette="husl")
for item in A1_PCAPlot.get_xticklabels():
    item.set_rotation(90)
plt.show()

""" PCA2 """

forest.fit(PCA2, y2)
importances = forest.feature_importances_
indices = np.argsort(importances )[::-1] 

A2_PCA = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    A2_PCA.iloc[count, 0] = PCA_Labels[i]
    A2_PCA.iloc[count, 1] = (importances[count])
    count = count +1
    
A2_PCA.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("RND-Forest PCA2")
print(A2_PCA)

A2_PCA = A2_PCA.iloc[:40] 
A2_PCAPlot =sns.barplot(x=A2_PCA["Name"], y=A2_PCA["Wert"], palette="husl")
for item in A2_PCAPlot.get_xticklabels():
    item.set_rotation(90)
plt.show()


# In[3] Gradient Boost

""" X1 """

forestBoost.fit(X1 , y1) 
importances = forestBoost.feature_importances_
indices = np.argsort(importances )[::-1] 

B1 = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    B1.iloc[count, 0] = X1_column_names[i]
    B1.iloc[count, 1] = (importances[count])
    count = count +1
    
B1.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("Gradient Boost X1")
print(B1)

B1 = B1.iloc[:40] 
B1Plot =sns.barplot(x=B1["Name"], y=B1["Wert"], palette="husl")
for item in B1Plot.get_xticklabels():
    item.set_rotation(90)
plt.show()


""" X2 """

forestBoost.fit(X2, y2) 
importances = forestBoost.feature_importances_
indices = np.argsort(importances )[::-1] 

B2 = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    B2.iloc[count, 0] = X1_column_names[i]
    B2.iloc[count, 1] = (importances[count])
    count = count +1
    
B2.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("Gradient Boost X2")
print(B2)

B2 = B2.iloc[:40] 
B2Plot =sns.barplot(x=B2["Name"], y=B2["Wert"], palette="husl")
for item in B2Plot.get_xticklabels():
    item.set_rotation(90)
plt.show()

""" PCA1 """

forestBoost.fit(PCA1 , y1) 
importances = forestBoost.feature_importances_
indices = np.argsort(importances )[::-1] 

B1_PCA = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    B1_PCA.iloc[count, 0] = PCA_Labels[i]
    B1_PCA.iloc[count, 1] = (importances[count])
    count = count +1
    
B1_PCA.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("Gradient Boost PCA1")
print(B1_PCA)

B1_PCA = B1_PCA.iloc[:40] 
B1_PCAPlot =sns.barplot(x=B1_PCA["Name"], y=B1_PCA["Wert"], palette="husl")
for item in B1_PCAPlot.get_xticklabels():
    item.set_rotation(90)
plt.show()

""" PCA2 """

forestBoost.fit(PCA2, y2)
importances = forestBoost.feature_importances_
indices = np.argsort(importances )[::-1] 

B2_PCA = pd.DataFrame(np.empty([len(indices), 2]), columns = ["Name", "Wert"])
count = 0

for i in indices:
    B2_PCA.iloc[count, 0] = PCA_Labels[i]
    B2_PCA.iloc[count, 1] = (importances[count])
    count = count +1
    
B2_PCA.sort_values(by = ["Wert"], inplace = True, ascending = False)
print("Gradient Boost PCA2")
print(B2_PCA)

B2_PCA = B2_PCA.iloc[:40] 
B2_PCAPlot =sns.barplot(x=B2_PCA["Name"], y=B2_PCA["Wert"], palette="husl")
for item in B2_PCAPlot.get_xticklabels():
    item.set_rotation(90)
plt.show()


# In[4] Neronal Network

"""
Feature Importance not possible
"""


# In[5]: SVM

"""
Feature Importance not possible
"""


# In[6] KNN

"""
Feature Importance not possible
"""


# In[7] Assign features more important than mean, median and the best 8 
#[ Attention, only from the best 40]

A1_mean = A1.where(A1.Wert > A1.Wert.mean())
A1_mean.to_csv("RF1_mean.csv")
A1_median = A1.where(A1.Wert > A1.Wert.median())
A1_median.to_csv("RF1_median.csv")
A1_8 = A1.iloc[:8]
A1_8.to_csv("RF1_8.csv")

A2_mean = A2.where(A2.Wert > A2.Wert.mean())
A2_mean.to_csv("RF2_mean.csv")
A2_median = A2.where(A2.Wert > A2.Wert.median())
A2_median.to_csv("RF2_median.csv")
A2_8 = A2.iloc[:8]
A2_8.to_csv("RF2_8.csv")


B1_mean = B1.where(B1.Wert > B1.Wert.mean())
B1_mean.to_csv("RBF1_mean.csv")
B1_median = B1.where(B1.Wert > B1.Wert.median())
B1_median.to_csv("RBF1_median.csv")
B1_sum = B1.iloc[:8]
B1_sum.to_csv("RBF1_8.csv")

B2_mean = B2.where(B2.Wert > B2.Wert.mean())
B2_mean.to_csv("RBF2_mean.csv")
B2_median = B2.where(B2.Wert > B2.Wert.median())
B2_median.to_csv("RBF2_median.csv")
B2_sum = B2.iloc[:8]
B2_sum.to_csv("RBF2_8.csv")
