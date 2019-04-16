# In[1]: Library import

import numpy as np
import pandas as pd
import sys
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
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

features1RF_mean = pd.read_csv(r"Data\RF1_mean.csv", header = 0).dropna()
features1RF_mean.columns = ["Ort","Name", "Wert"]
features1RF_median = pd.read_csv(r"Data\RF1_median.csv", header = 0).dropna()
features1RF_median.columns = ["Ort","Name", "Wert"]
features1RF_8 = pd.read_csv(r"Data\RF1_8.csv", header = 0).dropna()
features1RF_8.columns = ["Ort","Name", "Wert"]

features2RF_mean = pd.read_csv(r"Data\RF2_mean.csv", header = 0).dropna()
features2RF_mean.columns = ["Ort","Name", "Wert"]
features2RF_median = pd.read_csv(r"Data\RF2_median.csv", header = 0).dropna()
features2RF_median.columns = ["Ort","Name", "Wert"]
features2RF_8 = pd.read_csv(r"Data\RF2_8.csv", header = 0).dropna()
features2RF_8.columns = ["Ort","Name", "Wert"]

features1RBF_mean = pd.read_csv(r"Data\RBF1_mean.csv", header = 0).dropna()
features1RBF_mean.columns = ["Ort","Name", "Wert"]
features1RBF_median = pd.read_csv(r"Data\RBF1_median.csv", header = 0).dropna()
features1RBF_median.columns = ["Ort","Name", "Wert"]
features1RBF_8 = pd.read_csv(r"Data\RBF1_8.csv", header = 0).dropna()
features1RBF_8.columns = ["Ort","Name", "Wert"]

features2RBF_mean = pd.read_csv(r"Data\RBF2_mean.csv", header = 0).dropna()
features2RBF_mean.columns = ["Ort","Name", "Wert"]
features2RBF_median = pd.read_csv(r"Data\RBF2_median.csv", header = 0).dropna()
features2RBF_median.columns = ["Ort","Name", "Wert"]
features2RBF_8 = pd.read_csv(r"Data\RBF2_8.csv", header = 0).dropna()
features2RBF_8.columns = ["Ort","Name", "Wert"]

endzeitDaten = timeit.default_timer()
print("Zeit bis die Daten eingelesen sind: ",endzeitDaten-startzeitDaten)


# In[3] Edit data 

startEd = timeit.default_timer()

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

y2_0,X2_0 =    resample(y2[y2==0], X2_imp[y2==0], replace = True, n_samples = 55, random_state = 0)
y2_1,X2_1 =    resample(y2[y2==1], X2_imp[y2==1], replace = True, n_samples = 55, random_state = 0)
y2_2,X2_2 =    resample(y2[y2==2], X2_imp[y2==2], replace = True, n_samples = 55, random_state = 0)
y2_3,X2_3 =    resample(y2[y2==3], X2_imp[y2==3], replace = True, n_samples = 55, random_state = 0)
y2_4,X2_4 =    resample(y2[y2==4], X2_imp[y2==4], replace = True, n_samples = 55, random_state = 0)
y2_5,X2_5 =    resample(y2[y2==5], X2_imp[y2==5], replace = True, n_samples = 55, random_state = 0)
y2_6,X2_6 =    resample(y2[y2==6], X2_imp[y2==6], replace = True, n_samples = 55, random_state = 0)
y2_7,X2_7 =    resample(y2[y2==7], X2_imp[y2==7], replace = True, n_samples = 55, random_state = 0)
y2_8,X2_8 =    resample(y2[y2==8], X2_imp[y2==8], replace = True, n_samples = 55, random_state = 0)
y2_9,X2_9 =    resample(y2[y2==9], X2_imp[y2==9], replace = True, n_samples = 55, random_state = 0)
y2_10,X2_10 =resample(y2[y2==10], X2_imp[y2==10], replace = True, n_samples = 55, random_state = 0)
y2_11,X2_11 =resample(y2[y2==11], X2_imp[y2==11], replace = True, n_samples = 55, random_state = 0)
y2_12,X2_12 =resample(y2[y2==12], X2_imp[y2==12], replace = True, n_samples = 55, random_state = 0)
y2_13,X2_13 =resample(y2[y2==13], X2_imp[y2==13], replace = True, n_samples = 55, random_state = 0)
y2_14,X2_14 =resample(y2[y2==14], X2_imp[y2==14], replace = True, n_samples = 55, random_state = 0)
y2_15,X2_15 =resample(y2[y2==15], X2_imp[y2==15], replace = True, n_samples = 55, random_state = 0)
y2_16,X2_16 =resample(y2[y2==16], X2_imp[y2==16], replace = True, n_samples = 55, random_state = 0)

list_namesX = [X2_1, X2_2, X2_3, X2_4, X2_5, X2_6, X2_7, X2_8, X2_9, X2_10,
               X2_11, X2_12, X2_13, X2_14, X2_15, X2_16]
list_namesy = [y2_1, y2_2, y2_3, y2_4, y2_5, y2_6, y2_7, y2_8, y2_9, y2_10,
               y2_11, y2_12, y2_13, y2_14, y2_15, y2_16]

X2_re = X2_0
for i in list_namesX:
    X2_re = X2_re.append(i)

y2_re = pd.DataFrame(y2_0, columns = ["splticrm"])
for i in list_namesy: 
    i = pd.DataFrame(i, columns = ["splticrm"])
    y2_re = y2_re.append(i)
    
rebaEnd = timeit.default_timer()
print("Rebalance zeit : ", rebaEnd-rebaStart)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_re ,y1_re ,test_size = 0.2, train_size = 300)#, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_re, y2_re, test_size = 0.2, train_size = 300)#, random_state = 0)

sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X2_train = sc.fit_transform(X2_train)

PCA1_train = PCA(n_components = 8).fit(X1_train).transform(X1_train)
PCA2_train = PCA(n_components = 8).fit(X2_train).transform(X2_train)

y1_train = np.array(y1_train).ravel()
y2_train = np.array(y2_train).ravel()

forest = RandomForestClassifier(max_depth = None)
forestBoost = GradientBoostingClassifier(max_depth = None)
MLP = MLPClassifier()
svm = SVC()
knn = KNeighborsClassifier()

Names = np.vstack((Names1, Names2))
Names = pd.DataFrame(Names, columns = ["Name", "ExName"])
mydict = dict(zip(Names.Name, Names.ExName))

X1_columns = pd.DataFrame(X1_column_names, columns = ["Name"])
X1_columns = X1_columns.replace(mydict)
X1_column_names = pd.DataFrame(X1_column_names)
Named = np.hstack((X1_column_names, X1_columns))

Named = pd.DataFrame(Named , columns = ["Name", "ExName"])

endEd = timeit.default_timer()
print("Edit time: ", endEd-startEd)


# In[4] Feature Selection

startFeat = timeit.default_timer()

DelahuntyAndOcallagan = [7, 16, 22, 41,59]  #Missing: retained eraning/total assets, interest coverage, total assets
                                            #Replaced (out-in): net Margin - Net Profit margin, marked to book - book to marked, return on total assets - roa, operating margin - operating marging bevor & after decap
DelahuntyAndOcallagan_names = Named.iloc[DelahuntyAndOcallagan]

HuangEtAl = [17,18,23,41,54]    #Missing: total assets, total liabilities,
                                #Replaced: log term debt/total capital - total dept/equity
HuangEtAl_names = Named.iloc[HuangEtAl]

Kim = [17, 18, 22, 41, 47, 54, 59, 65, 67]  #Missing: Total Assets, interest coverage
                                            #Replaced: sales/fixed assets - sales/ invested capital & sales/working capital, operating margin - operating marging bevor & after decap, log term debt/total capital - total dept/equity, CF/current liabilities - op CF/current liabilities
Kim_names = Named.iloc[Kim]

BarbazonAndONeill = [7, 16, 22, 41, 59]     #Missing: Retained earnings/total assets, interest coverage, total assets
                                            #Replaced: net Margin - Net Profit margin, marked to book - book to marked, return on total assets - roa
BarbazonAndONeill_names = Named.iloc[BarbazonAndONeill]

HwangEtAl = [8, 16, 17, 18, 21, 22, 23, 24, 42, 43, 53]     #Missing: KMV-Morton_defaoult-proba, Earnings, total assets, tot. assets/equity, EBITDA/interests, Intersts, retained earnigs/ total assets
                                                            #Replaced: Market Equity Value - enterprise value multiple, long term debt/ total capital - total debt/total capital, net income - net profit margin, operating margin - operating marging bevor & after decap
HwangEtAl_names = Named.iloc[HwangEtAl]

range_ = (range(len(X2_column_names)))
random_ = random.randint(0,1234567890987)
random.seed(random_)
rand1 = random.sample(range_, 7)
rand1_names = Named.iloc[rand1]
rand2 = random.sample(range_, 7)
rand1_names = Named.iloc[rand2]
rand3 = random.sample(range_, 7)
rand3_names = Named.iloc[rand3]
rand4 = random.sample(range_, 7)
rand4_names = Named.iloc[rand4]
rand5 = random.sample(range_, 7)
rand5_names = Named.iloc[rand5]

ownD = [74, 73, 35, 41, 21, 58 ]
ownD_names = Named.iloc[ownD]

alles = np.arange(start = 0, stop = len(X2_column_names)+1, step = 1)

Features_list = [DelahuntyAndOcallagan, HuangEtAl, Kim, BarbazonAndONeill,
                 HwangEtAl, rand1, rand2, rand3, rand4, rand5, ownD,
                 features1RF_mean.Ort, features1RF_median.Ort, features1RF_8.Ort, 
                 features2RF_mean.Ort, features2RF_median.Ort, features2RF_8.Ort, 
                 features1RBF_mean.Ort, features1RBF_median.Ort, features1RBF_8.Ort, 
                 features2RBF_mean.Ort, features2RBF_median.Ort, features2RBF_8.Ort
                 #,alles
                 ]
Names_list = ["DelahuntyAndOcallagan", "HuangEtAl", "Kim", "BarbazonAndONeill",
              "HwangEtAl", "rand1", "rand2", "rand3", "rand4", "rand5", "ownD",
              "features1RF_mean", "features1RF_median", "features1RF_8", 
              "features2RF_mean", "features2RF_median", "features2RF_8", 
              "features1RBF_mean", "features1RBF_median", "features1RBF_8", 
              "features2RBF_mean", "features2RBF_median", "features2RBF_8"
              #,"alles"
             ]

SolutionArray = np.zeros((len(Features_list)*7, 4))
SolutionFrame = pd.DataFrame(SolutionArray, columns = ["X1", "X2", "ListName", "Number"])
c = 0

endFeat = timeit.default_timer()
print("Feature Selection time: ", endFeat-startFeat)


# In[5] : 1. GridSearch RND_Forest

#r"""
RFGridStart = timeit.default_timer()

_n_estimators = np.array([1,10,50,100,200,400,800])
_criterion = np.array(["gini", "entropy"])
_max_depth = np.array([1,10,50,100,200,400,800])
_max_features = pd.DataFrame(["auto", "log2", None])
_bootstrap = np.array([True, False])
_oob_score = np.array([True, False]) #only if bootstrap = true --> oob_score = true 
_n_jobs = [-1]
_random_state = np.arange(start = 0, stop = 1000, step = 13)
_warm_start = np.array([True, False])

rnd_forest_grid = {"n_estimators": _n_estimators,
                 "criterion": _criterion,
                 #"max_depht": _max_depht, 
                 #"max_features": _max_features,
                 "bootstrap": _bootstrap,
                 #"oob_score": _oob_score,
                 #"n_jobs": _n_jobs,
                 #"random_state": _random_state,
                 "warm_start": _warm_start 
                 }  

grid_search1RF = GridSearchCV(estimator = forest, param_grid = rnd_forest_grid, cv = 3, n_jobs = -1)

RFGridFit = timeit.default_timer()

grid1RF_PCAfit1 = grid_search1RF.fit(PCA1_train, y1_train) 
grid1RF_PCAfit2 = grid_search1RF.fit(PCA2_train, y2_train) 

print("Grid 1 PCA: ", grid1RF_PCAfit1.best_params_, " ", grid1RF_PCAfit1.best_score_ ,
"Grid 2 PCA: ", grid1RF_PCAfit2.best_params_, " ", grid1RF_PCAfit2.best_score_)

c1 = c
for i in Features_list:
    SolutionFrame.iloc[c,0] = grid_search1RF.fit(X1_train[:, i], y1_train)
    print(Names_list[c%len(Names_list)] ,"X1: ", SolutionFrame.iloc[c,0].best_score_,
                     SolutionFrame.iloc[c,0].best_params_ )
    
    SolutionFrame.iloc[c,1] = grid_search1RF.fit(X2_train[:, i], y2_train)
    print(Names_list[c%len(Names_list)] ,"X2: ", SolutionFrame.iloc[c,1].best_score_,
                     SolutionFrame.iloc[c,1].best_params_ )
    
    SolutionFrame.iloc[c,2] = Names_list[c%len(Names_list)]
    SolutionFrame.iloc[c,3] = c

    c = c+1


RFGend = timeit.default_timer()
print("Time 4 :", RFGend-RFGridFit)
print("RF total: ", RFGend- RFGridStart)
#"""


# In[6] GridSearch GradientBoostingClassifier

#r"""
GradStart = timeit.default_timer()

_loss = ["deviance", "exponential"]
_learning_rate = np.linspace(start = 1e-4, stop = 1, num = 10)
_n_estimators = [10,50,100,200,800]
_subsample = np.linspace(start = 0.1, stop = 1.0, num = 3)
_max_depth = np.arange(start = 1, stop = 10, step = 2)
_min_impurity_split = np.linspace(start = 1e-9, stop = 1e-4, num = 10)
_warm_start = [True, False]
_tol = np.linspace(start = 1e-8, stop = 1e-2, num = 10)

GradientGrid = { #"loss": _loss,
                 "learning_rate": _learning_rate,
                 "subsample": _subsample,
                 "max_depth": _max_depth,
                 "min_impurity_split": _min_impurity_split, 
                 "warm_start": _warm_start,
                 "tol": _tol
                 }

GridSearchGB = GridSearchCV(estimator = forestBoost, param_grid = GradientGrid, cv = 3, n_jobs = -1)

GradInitialize = timeit.default_timer()
print("Grad time 1:", GradInitialize-GradStart)

gridgrad_PCAfit1 = GridSearchGB.fit(PCA1_train, y1_train)
gridgrad_PCAfit2 = GridSearchGB.fit(PCA2_train, y2_train)

GradEnd = timeit.default_timer()
print("Grad Time 2: ",GradEnd-GradInitialize)

print("Grid 1 PCA: ", gridgrad_PCAfit1.best_params_, " ", gridgrad_PCAfit1.best_score_ ,
      "Grid 2 PCA: ", gridgrad_PCAfit2.best_params_, " ", gridgrad_PCAfit2.best_score_)



c2 = c
for i in Features_list:
    SolutionFrame.iloc[c,0] = GridSearchGB.fit(X1_train[:, i], y1_train)
    print(Names_list[c%len(Names_list)] ,"X1: ", SolutionFrame.iloc[c,0].best_score_,
                     SolutionFrame.iloc[c,0].best_params_ )
    
    SolutionFrame.iloc[c,1] = GridSearchGB.fit(X2_train[:, i], y2_train)
    print(Names_list[c%len(Names_list)] ,"X2: ", SolutionFrame.iloc[c,1].best_score_,
                     SolutionFrame.iloc[c,1].best_params_ )
    
    SolutionFrame.iloc[c,2] = Names_list[c%len(Names_list)]
    SolutionFrame.iloc[c,3] = c

    c = c+1


GradEndd = timeit.default_timer()
print("Grad Total : ", GradEndd-GradStart)
#"""


# In[7] 2. GridSearch Neural Network

#r"""
NNetStart = timeit.default_timer()   

_hidden_layer_sizes = [(1,), (10,), (100,), (500,)]
_activation = ['identity', 'logistic', 'tanh', 'relu']
_solver = ['lbfgs', 'sgd', 'adam']
_alpha = np.linspace(start = 0.0001, stop = 0.01, num = 10)
_learning_rate = ['constant', 'invscaling', 'adaptive']
_power_t = np.linspace(start = 0.0, stop = 1.0, num = 10)
_random_state = np.arange(start = 0, stop = 1000, step = 42)
_tol = np.linspace(start = 1e-8, stop = 1e-1, num = 10)
_warm_start = [True, False]
_momentum = np.linspace(start = 0.5, stop = 1.0, num = 10)
_early_stopping = [True]
_validation_fraction = np.linspace(start = 0.01, stop = 0.5, num = 10)
_beta_1 = np.linspace(start = 0.5, stop = 1.0, num = 10)
_beta_2 = np.linspace(start = 0.9, stop = 1.0, num = 10) 
_epsilon = np.linspace(start = 0.000000001, stop = 0.00000001, num = 10)

NNetGrid = {"hidden_layer_sizes": _hidden_layer_sizes,
            "activation": _activation,
            "solver": _solver,
            "alpha": _alpha,
            "learning_rate": _learning_rate,
            "power_t": _power_t,
           # "random_state": _random_state,
            "tol": _tol,
            "warm_start": _warm_start,
            "momentum": _momentum,
            "early_stopping": _early_stopping,
            "validation_fraction": _validation_fraction,
            #"beta_1": _beta_1,
            #"_beta_2": _beta_2,
            #"epsilon": _epsilon 
            }

NNetInitialize = timeit.default_timer()
print("NNet Time 1:", NNetInitialize-NNetStart)

grid_searchNN = GridSearchCV(estimator = MLP, param_grid = NNetGrid, cv = 3, n_jobs = -1)

gridNNet_PCAfit1 = grid_searchNN.fit(PCA1_train, y1_train)
gridNNet_PCAfit2 = grid_searchNN.fit(PCA2_train, y2_train)

NNetFit = timeit.default_timer()
print("NNet Time 2: ", NNetFit-NNetInitialize)

print("Grid 1 PCA: ", gridNNet_PCAfit1.best_params_, " ", gridNNet_PCAfit1.best_score_ ,
"Grid 2 PCA: ", gridNNet_PCAfit2.best_params_, " ", gridNNet_PCAfit2.best_score_)

c3 = c
for i in Features_list:
    SolutionFrame.iloc[c,0] = grid_searchNN.fit(X1_train[:, i], y1_train)
    print(Names_list[c%len(Names_list)] ,"X1: ", SolutionFrame.iloc[c,0].best_score_,
                     SolutionFrame.iloc[c,0].best_params_ )
    
    SolutionFrame.iloc[c,1] = grid_searchNN.fit(X2_train[:, i], y2_train)
    print(Names_list[c%len(Names_list)] ,"X2: ", SolutionFrame.iloc[c,1].best_score_,
                     SolutionFrame.iloc[c,1].best_params_ )
    
    SolutionFrame.iloc[c,2] = Names_list[c%len(Names_list)]
    SolutionFrame.iloc[c,3] = c

    c = c+1


NNetEnd = timeit.default_timer()
print("NNet Time 3: ", NNetEnd-NNetFit)
print("NNet Total: ", NNetEnd-NNetStart)
#"""


# In[8] GridSearch SVM

#r"""
svmStart = timeit.default_timer()

_C = np.linspace(start = 1e-10, stop = 2, num = 10)
_kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
_degree = np.arange(start= 1, stop = 100, step = 10)
_gamma = np.linspace(start = 0.0, stop = 0.00001, num = 10)
_coef0 = np.linspace(start = 0.0, stop = 1.0, num = 10)
_shrinking = [True, False]
_probability = [True, False]
_tol = np.linspace(start = 1e-8, stop = 1e-1, num = 10)
_decision_function_shape = ["ovo", "ovr"]

svmGrid = {"C" : _C, 
           "kernel": _kernel,
           "degree": _degree,
           "gamma": _gamma, 
           "coef0": _coef0,
           "shrinking": _shrinking,
           "probability": _probability, 
           "tol": _tol, 
           "decision_function_shape": _decision_function_shape
           }

GridSearchSVM = GridSearchCV(estimator = svm, param_grid = svmGrid, cv = 3, n_jobs = -1)

svminitialize = timeit.default_timer()
print("SVM Zeit 1: ", svminitialize-svmStart)

svmGrid_PCAfit1 = GridSearchSVM.fit(PCA1_train, y1_train)
svmGrid_PCAfit2 = GridSearchSVM.fit(PCA2_train, y2_train)

svmend = timeit.default_timer()
print("svm Zeit 2 : ", svmend-svminitialize)

print("Grid 1 PCA: ", svmGrid_PCAfit1.best_params_, " ", svmGrid_PCAfit1.best_score_ ,
 "Grid 1 PCA: ", svmGrid_PCAfit2.best_params_, " ", svmGrid_PCAfit2.best_score_)
c4 = c
for i in Features_list:
    SolutionFrame.iloc[c,0] = GridSearchSVM.fit(X1_train[:, i], y1_train)
    print(Names_list[c%len(Names_list)] ,"X1: ", SolutionFrame.iloc[c,0].best_score_,
                     SolutionFrame.iloc[c,0].best_params_ )
    
    SolutionFrame.iloc[c,1] = GridSearchSVM.fit(X2_train[:, i], y2_train)
    print(Names_list[c%len(Names_list)] ,"X2: ", SolutionFrame.iloc[c,1].best_score_,
                     SolutionFrame.iloc[c,1].best_params_ )
    
    SolutionFrame.iloc[c,2] = Names_list[c%len(Names_list)]
    SolutionFrame.iloc[c,3] = c

    c = c+1

svmendd = timeit.default_timer()

print("SVM total time : ", svmendd-svmStart)  
#"""

    
# In[9] GridSearch K-NearestNeigbors

#r"""
knnstart = timeit.default_timer()

_n_eighbors = np.arange(start = 3, stop = len(X1_train), step = 2) #nur ungerade zahlen sonst indifferent
_weights = ["uniform", "distance"]
_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'] 
_leaf_size = [1, 10, 30, 70, 200]
_p = np.arange(start = 1, stop = 10, step = 1)
#_metric if needed

KnnGrid = {"n_eighbors": _n_eighbors, 
           "weights": _weights, 
           "algorithm": _algorithm, 
           "leaf_size": _leaf_size,
           "p": _p
           }

knnInitialize = timeit.default_timer()
print("Knn time 1: ", knnstart-knnInitialize)

GridSearchKNN = GridSearchCV(estimator = knn, param_grid = KnnGrid, cv = 3, n_jobs = -1)

knnGrid_PCAfit1 = GridSearchKNN.fit(PCA1_train, y1_train)
knnGrid_PCAfit2 = GridSearchKNN.fit(PCA2_train, y2_train)

knnEnd = timeit.default_timer()
print("knn zeit 2 : ", knnEnd-knnInitialize)

print("Grid 1 PCA: ", knnGrid_PCAfit1.best_params_, " ", knnGrid_PCAfit1.best_score_ ,
      "Grid 2 PCA: ", knnGrid_PCAfit2.best_params_, " ", knnGrid_PCAfit2.best_score_)
c5 = c
for i in Features_list:
    SolutionFrame.iloc[c,0] = GridSearchKNN.fit(X1_train[:, i], y1_train)
    print(Names_list[c%len(X2_column_names)] ,"X1: ", SolutionFrame.iloc[c,0].best_score_,
                     SolutionFrame.iloc[c,0].best_params_ )
    
    SolutionFrame.iloc[c,1] = GridSearchKNN.fit(X2_train[:, i], y2_train)
    print(Names_list[c%len(X2_column_names)] ,"X2: ", SolutionFrame.iloc[c,1].best_score_,
                     SolutionFrame.iloc[c,1].best_params_ )
    
    SolutionFrame.iloc[c,2] = Names_list[c%len(X2_column_names)]
    SolutionFrame.iloc[c,3] = c

    c = c+1


knnEndd = timeit.default_timer()
print("knn total time: ", knnEndd-knnstart)
#"""


# In[10] Pipelines

r"""
if __name__ == "__main__":
    
    pipeKNN_RF1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest ) ])
    pipeKNN_RF2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest ) ])
    pipeKNN_RF_PCA1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest ) ])
    pipeKNN_RF_PCA2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest ) ])
    pipeKNN_BRF1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost ) ])
    pipeKNN_BRF2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost ) ])
    pipeKNN_BRF_PCA1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost ) ])
    pipeKNN_BRF_PCA2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost ) ])
    pipeSVM_RF1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest) ])
    pipeSVM_RF2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest) ])
    pipeSVM_RF_PCA1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest) ])
    pipeSVM_RF_PCA2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forest) ])
    pipeSVM_BRF1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost) ])
    pipeSVM_BRF2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost) ])
    pipeSVM_BRF_PCA1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost) ])
    pipeSVM_BRF_PCA2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", forestBoost) ])

    RF_KNN_fit1 = pipeKNN_RF1.fit(X1_train, y1_train) 
    RF_KNN_fit2 = pipeKNN_RF2.fit(X2_train, y2_train) 
    BRF_KNN_fit1 = pipeKNN_BRF1.fit(X1_train, y1_train) 
    BRF_KNN_fit2 = pipeKNN_BRF2.fit(X2_train, y2_train) 
    RF_SVM_fit1 = pipeSVM_RF1.fit(X1_train, y1_train) 
    RF_SVM_fit2 = pipeSVM_RF2.fit(X2_train, y2_train) 
    RF_SVM_fit1 = pipeSVM_BRF1.fit(X1_train, y1_train) 
    RF_SVM_fit2 = pipeSVM_BRF2.fit(X2_train, y2_train) 
    
    RF_KNN_fitPCA1 = pipeKNN_RF_PCA1.fit(PCA1_train, y1_train) 
    RF_KNN_fitPCA2 = pipeKNN_RF_PCA2.fit(PCA2_train, y2_train) 
    BRF_KNN_fitPCA1 = pipeKNN_BRF_PCA1.fit(PCA1_train, y1_train) 
    BRF_KNN_fitPCA2 = pipeKNN_BRF_PCA2.fit(PCA2_train, y2_train) 
    RF_SVM_fitPCA1 = pipeSVM_RF_PCA1.fit(PCA1_train, y1_train) 
    RF_SVM_fitPCA2 = pipeSVM_RF_PCA2.fit(PCA2_train, y2_train) 
    RF_SVM_fitPCA1 = pipeSVM_BRF_PCA1.fit(PCA1_train, y1_train) 
    RF_SVM_fitPCA2 = pipeSVM_BRF_PCA2.fit(PCA2_train, y2_train) 


    pipeKNN_NN1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP ) ])
    pipeKNN_NN2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP ) ])
    pipeKNN_NN_PCA1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP ) ])
    pipeKNN_NN_PCA2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP ) ])
    pipeSVM_NN1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP) ])
    pipeSVM_NN2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP) ])
    pipeSVM_NN_PCA1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP) ])
    pipeSVM_NN_PCA2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", MLP) ])

    NN_KNN_fit1 = pipeKNN_NN1.fit(X1_train, y1_train) 
    NN_KNN_fit2 = pipeKNN_NN2.fit(X2_train, y2_train) 
    NN_KNN_fit_PCA1 = pipeKNN_NN_PCA1.fit(PCA1_train, y1_train) 
    NN_KNN_fit_PCA2 = pipeKNN_NN_PCA2.fit(PCA2_train, y2_train) 

    NN_SVM_fit1 = pipeSVM_NN1.fit(X1_train, y1_train) 
    NN_SVM_fit2 = pipeSVM_NN2.fit(X2_train, y2_train) 
    NN_SVM_fit_PCA1 = pipeSVM_NN_PCA1.fit(PCA1_train, y1_train) 
    NN_SVM_fit_PCA2 = pipeSVM_NN_PCA2.fit(PCA2_train, y2_train) 


    pipeKNN_SVM1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm ) ])
    pipeKNN_SVM2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm ) ])
    pipeKNN_SVM_PCA1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm ) ])
    pipeKNN_SVM_PCA2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm ) ])
    pipeSVM_SVM1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm) ])
    pipeSVM_SVM2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm) ])
    pipeSVM_SVM_PCA1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm) ])
    pipeSVM_SVM_PCA2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", svm) ])

    SVM_KNN_fit1 = pipeKNN_SVM1.fit(X1_train, y1_train) 
    SVM_KNN_fit2 = pipeKNN_SVM2.fit(X2_train, y2_train) 
    SVM_KNN_fit_PCA1 = pipeKNN_SVM_PCA1.fit(PCA1_train, y1_train) 
    SVM_KNN_fit_PCA2 = pipeKNN_SVM_PCA2.fit(PCA2_train, y2_train) 

    SVM_SVM_fit1 = pipeSVM_SVM1.fit(X1_train, y1_train) 
    SVM_SVM_fit2 = pipeSVM_SVM2.fit(X2_train, y2_train) 
    SVM_SVM_fit_PCA1 = pipeSVM_SVM_PCA1.fit(PCA1_train, y1_train) 
    SVM_SVM_fit_PCA2 = pipeSVM_SVM_PCA2.fit(PCA2_train, y2_train) 


    pipeKNN_KNN1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn ) ])
    pipeKNN_KNN2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn ) ])
    pipeKNN_KNN_PCA1 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn ) ])
    pipeKNN_KNN_PCA2 = Pipeline([("feature_selection", SelectFromModel(knnGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn ) ])
    pipeSVM_KNN1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn) ])
    pipeSVM_KNN2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_fit2)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn) ])
    pipeSVM_KNN_PCA1 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn) ])
    pipeSVM_KNN_PCA2 = Pipeline([("feature_selection", SelectFromModel(svmGrid_PCAfit1)),
                        ("scaler", StandardScaler()),
                        ("Classifier", knn) ])

    KNN_KNN_fit1 = pipeKNN_KNN1.fit(X1_train, y1_train) 
    KNN_KNN_fit2 = pipeKNN_KNN2.fit(X2_train, y2_train) 
    KNN_KNN_fit_PCA1 = pipeKNN_KNN_PCA1.fit(PCA1_train, y1_train) 
    KNN_KNN_fit_PCA2 = pipeKNN_KNN_PCA2.fit(PCA2_train, y2_train) 

    KNN_SVM_fit1 = pipeSVM_KNN1.fit(X1_train, y1_train) 
    KNN_SVM_fit2 = pipeSVM_KNN2.fit(X2_train, y2_train) 
    KNN_SVM_fit_PCA1 = pipeSVM_KNN_PCA1.fit(PCA1_train, y1_train) 
    KNN_SVM_fit_PCA2 = pipeSVM_KNN_PCA2.fit(PCA2_train, y2_train) 
#"""    
