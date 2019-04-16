# In[1]: Library import

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

import timeit

# In[2]: Import data

startzeitDaten = timeit.default_timer()

data = pd.read_csv(r"Data\SP500_data_new.csv")#,
                 #  parse_dates = ["datadate","adate", "qdate"], dayfirst = True)#, index_col=["gvkey", "datadate"]) 
data_NaN = data.dropna()
data_y = data.dropna(subset =["splticrm"])

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
X2 = np.nan_to_num(X2)
X2 = pd.DataFrame(X2, columns = X2_column_names )

corr_frame1 = X1.assign(splticrm = y1)
corr_frame2 = X2.assign(splticrm = y2)

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


y2_0,X2_0 =resample(y2[y2==0], X2[y2==0], replace = True, n_samples = 55, random_state = 0)
y2_1,X2_1 =resample(y2[y2==1], X2[y2==1], replace = True, n_samples = 55, random_state = 0)
y2_2,X2_2 =resample(y2[y2==2], X2[y2==2], replace = True, n_samples = 55, random_state = 0)
y2_3,X2_3 =resample(y2[y2==3], X2[y2==3], replace = True, n_samples = 55, random_state = 0)
y2_4,X2_4 =resample(y2[y2==4], X2[y2==4], replace = True, n_samples = 55, random_state = 0)
y2_5,X2_5 =resample(y2[y2==5], X2[y2==5], replace = True, n_samples = 55, random_state = 0)
y2_6,X2_6 =resample(y2[y2==6], X2[y2==6], replace = True, n_samples = 55, random_state = 0)
y2_7,X2_7 =resample(y2[y2==7], X2[y2==7], replace = True, n_samples = 55, random_state = 0)
y2_8,X2_8 =resample(y2[y2==8], X2[y2==8], replace = True, n_samples = 55, random_state = 0)
y2_9,X2_9 =resample(y2[y2==9], X2[y2==9], replace = True, n_samples = 55, random_state = 0)
y2_10,X2_10 =resample(y2[y2==10], X2[y2==10], replace = True, n_samples = 55, random_state = 0)
y2_11,X2_11 =resample(y2[y2==11], X2[y2==11], replace = True, n_samples = 55, random_state = 0)
y2_12,X2_12 =resample(y2[y2==12], X2[y2==12], replace = True, n_samples = 55, random_state = 0)
y2_13,X2_13 =resample(y2[y2==13], X2[y2==13], replace = True, n_samples = 55, random_state = 0)
y2_14,X2_14 =resample(y2[y2==14], X2[y2==14], replace = True, n_samples = 55, random_state = 0)
y2_15,X2_15 =resample(y2[y2==15], X2[y2==15], replace = True, n_samples = 55, random_state = 0)
y2_16,X2_16 =resample(y2[y2==16], X2[y2==16], replace = True, n_samples = 55, random_state = 0)
y2_17,X2_17 =resample(y2[y2==17], X2[y2==17], replace = True, n_samples = 55, random_state = 0)
y2_18,X2_18 =resample(y2[y2==18], X2[y2==18], replace = True, n_samples = 55, random_state = 0)

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

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_re ,y1_re,#to leave out resample just replace X1_re with X1 and y1_re with y1
                                                        test_size = 0.2, random_state = 0)#, train_size = 300)#
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_re, y2_re,#to leave out resample just replace X2_re with X2 and y2_re with y2
                                                        test_size = 0.2, random_state = 0)#, train_size = 300)#

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

endzeitDaten = timeit.default_timer()
print("Zeit bis die Daten eingelesen sind: ",endzeitDaten-startzeitDaten)


# In[2]:
#"""

startzeitCor = timeit.default_timer()

corr_NaN = corr_frame1.corr()
print("Corr_NaN: ", corr_NaN.splticrm)
#corr_NaN.splticrm.to_csv("Corr_NaN.csv")
fig1 = plt.figure()
plt.matshow(corr_NaN)
plt.title("Corr NaN")
#plt.show()
plt.close(fig1)

corr_y = corr_frame2.corr()
print("Corr y: ", corr_y.splticrm)
#corr_y.splticrm.to_csv("Corr_y.csv")
plt.matshow(corr_y)
plt.title("Corr y")
#plt.show()


endzeitCor = timeit.default_timer()

print("Cor-Zeit: ", endzeitCor-startzeitCor)
"""

# In[3]:
"""

startzeitregplot = timeit.default_timer()

counter = 0
for i in X1_column_names:
    stime = timeit.default_timer()    
    regfig = plt.figure()
    regfig = sns.regplot(x= X2[i], y=y2)
    plt.title(i)
    saver = i+"1"+".png"
    plt.savefig(saver)
    plt.close()
    counter = counter +1
    time = timeit.default_timer()
    print(counter)
    print("Zeit regplot1: ", time -stime)

counter = 0
for i in X2_column_names:
    stime2 = timeit.default_timer()    
    regfig = plt.figure()
    regfig = sns.regplot(x= X2[i], y=y2)
    plt.title(i)
    saver = i+"2"+".png"
    plt.savefig(saver)
    plt.close()
    counter = counter +1
    time2 = timeit.default_timer()
    print(counter)
    print("Zeit regplot2: ", time2 -stime2)
    
endzeitregplots = timeit.default_timer()

print("Zeit für alle regplots: ", endzeitregplots-startzeitregplot)
#"""



# In[4]:

#"""

plt.ioff()

startzeitPlots = timeit.default_timer()

plot_Nan = plt.figure()
plot_Nan = sns.pairplot( data = corr_frame1, x_vars = X1_column_names,
                        y_vars = ["splticrm"], size = 4 ,  dropna = True, hue = "gvkey", palette ="husl")
plt.title("pairplot1")
plot_Nan.savefig("Pairplot1.png")
plt.close()

plot_y = plt.figure()
plot_y = sns.pairplot( data = corr_frame2, x_vars = X1_column_names,
                      y_vars = ["splticrm"], size = 4 ,  dropna = True, hue = "gvkey", palette = "husl")
plt.title("pairplot2")
plot_y.savefig("Pairplot2.png")
plt.close()

endzeitPlots = timeit.default_timer()

print("PLotzeit: ", endzeitPlots-startzeitPlots)
#"""



# In[5]:
#"""

startzeitRF = timeit.default_timer()

#getauscht pca und rnd forest 
pipeRF = Pipeline([("feature_selection", SelectFromModel(RandomForestClassifier(), threshold = "median") ), ("scaler" ,StandardScaler()),("classification", SVC()) ])

#pipeRF = Pipeline([("feature_selection", SelectFromModel(PCA(n_components = 10))),(("scaler" ,StandardScaler()),("classification", (RandomForestClassifier(), threshold = "median") ) )])

forest = RandomForestClassifier( random_state=0, n_jobs = -1)#criterion ="mean"

forestBoost = GradientBoostingClassifier(n_estimators=100,max_features=0.5,random_state=1)

score1 = forest.fit(X1_train, y1_train).score(X1_test, y1_test)
score2 = forest.fit(X2_train, y2_train).score(X2_test, y2_test)

pipescore1 = pipeRF.fit(X1_train, y1_train).score(X1_test, y1_test)
pipescore2 = pipeRF.fit(X2_train, y2_train).score(X2_test, y2_test)

scoreboost1 = forestBoost.fit(X1_train, y1_train).score(X1_test, y1_test)
scoreboost2 = forestBoost.fit(X2_train, y2_train).score(X2_test, y2_test)

endzeitRF =timeit.default_timer()

print("Score 1: ", score1,"Score 2: ",  score2)
print(" ")
print("Pipe Score: ", "Pipe Score 1: ", pipescore1,"Pipe Score 2: ",  pipescore2)
print("Boost Score 1: ", scoreboost1, "Boost Score 2 : ", scoreboost2)
print(" ")

print("Zeit für den Random Forest Algorithmus: ", endzeitRF-startzeitRF)
#"""


# In[6]

"""Support Vector Maschines with BaggingClassifier and Boosting ---Braucht eine weile zum Rechnen---  """

#"""
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X2_train = scaler.fit_transform(X2_train)
X1_test = scaler.fit_transform(X1_test)
X2_test = scaler.fit_transform(X2_test)

startzeitSVM = timeit.default_timer()

svm_linear = SVC(kernel = "linear", random_state = 0, tol = 0.01)
svm_poly = SVC(kernel = "poly", random_state = 0, tol = 0.01)
svm_gaussian = SVC(random_state = 0, tol = 0.01)
#"""

#"""
svmzeit = timeit.default_timer()

svm_linear_fit1 = svm_linear.fit(X1_train, y1_train)
svm_linear_fit2 = svm_linear.fit(X2_train, y2_train)

svm1 = timeit.default_timer()
print("Zeit für svm linear fit: ", svm1-svmzeit)

svm_poly_fit1 = svm_poly.fit(X1_train, y1_train)
svm_poly_fit2 = svm_poly.fit(X2_train, y2_train)

svm2 = timeit.default_timer()
print("Zeit für svm poly fit: ", svm2-svm1)

svm_gaussian_fit1 = svm_gaussian.fit(X1_train, y1_train)
svm_gaussian_fit2 = svm_gaussian.fit(X2_train, y2_train)
svm4 = timeit.default_timer()
print("Zeit für gaussian fit: ", svm4-svm2)

score_linear1 = svm_linear_fit1.score(X1_test, y1_test)
score_linear2 = svm_linear_fit2.score(X2_test, y2_test)
svm3 = timeit.default_timer()
print("Zeit für linear score : ", svm3-svm2)

score_poly1 = svm_poly_fit1.score(X1_test, y1_test)
score_poly2 = svm_poly_fit2.score(X2_test, y2_test)
svm4 = timeit.default_timer()
print("Zeit für poly score : ", svm4-svm3)

score_gauss1 = svm_gaussian_fit1.score(X1_test, y1_test)
score_gauss2 = svm_gaussian_fit2.score(X2_test, y2_test)
svm7 = timeit.default_timer()
print("Zeit für gaussian score: ",svm7-svm4 )

svmendzeit = timeit.default_timer()
print("SVM zeit: ", svmendzeit -svmzeit)
print("Linear :", "Score 1: ", score_linear1, "Score 2: ", score_linear2, "Poly : ", "Score 1: ", score_poly1, "Score 2: ", score_poly2, "Score 1: ", score_gauss1, "Score 2 : ", score_gauss2)
print(" ")
#"""

# In[7]

"""Bagging """

#"""
startBag = timeit.default_timer()

model_linear = BaggingClassifier(base_estimator = svm_linear, n_estimators = 50, random_state = 1)
score_linear_bag1 = model_linear.fit(X1_train, y1_train).score(X1_test, y1_test)
score_linear_bag2 = model_linear.fit(X2_train, y2_train).score(X2_test, y2_test)
bag1 = timeit.default_timer()
print("linear bag score zeit : ", startBag-bag1)

model_poly = BaggingClassifier(base_estimator = svm_poly, n_estimators = 50, random_state = 1)
score_poly_bag1 = model_poly.fit(X1_train, y1_train).score(X1_test, y1_test)
score_poly_bag2 = model_poly.fit(X2_train, y2_train).score(X2_test, y2_test)
bag2 = timeit.default_timer()
print("poly bag score zeit", bag2-bag1)

model_gauss = BaggingClassifier(base_estimator = svm_gaussian, n_estimators = 50, random_state = 1)
score_gauss_bag1 = model_gauss.fit(X1_train, y1_train).score(X1_test, y1_test)
score_gauss_bag2 = model_gauss.fit(X2_train, y2_train).score(X2_test, y2_test)
bag3 = timeit.default_timer()
print("linear bag score zeit : ", bag3-bag2)

endbag = timeit.default_timer()
print("BagTime: ", endbag-startBag)
print("linear BagScore1: ", score_linear_bag1, "linear BagScore2 :", score_linear_bag2)
print("poly BagScoer1: ", score_poly_bag1, "poly Bagscore2: ", score_poly_bag2)

endzeitSVM = timeit.default_timer()

print("Zeit für den Support Vector Machines Algorithmus: ", endzeitSVM-startzeitSVM)

#"""

# In[8]

"""Boosting--- faltscher base estimator-- needs predict proba estimator---"""

#"""
boosttime = timeit.default_timer()

boostmodel_linear = AdaBoostClassifier(base_estimator = forest, random_state = 1)
boostmodelt = timeit.default_timer()
print("Boost model time: ", boosttime-boostmodelt)

boostscore_linear1 = boostmodel_linear.fit(X1_train, y1_train).score(X1_test, y1_test)
boostscore_linear2 = boostmodel_linear.fit(X2_train, y2_train).score(X2_test, y2_test)
boost1 = timeit.default_timer()
print("linear boost", boost1-boostmodelt)


boostend = timeit.default_timer()
print("Boosttime: ", boostend-boosttime)
print("Boostscore_linear1 : ", boostscore_linear1, "Boostscore_linear2 : ", boostscore_linear2)


#"""

# In[9]:


"""Pipeline with PCA ---Auch das geht eine weile--- """

#"""
startzeitPCA = timeit.default_timer()

pipe = Pipeline([("scaler ", StandardScaler ()), ("svm", SVC())])
parameter_grid = { "C " :[0.1 , 1, 10, 100], "degree " :[1, 2, 3, 5, 7]}
grid = GridSearchCV(pipe , param_grid=parameter_grid , cv=5, n_jobs=-1) 
gridtime = timeit.default_timer()
print("Gridsearch time : ", gridtime-startzeitPCA)

grid_fit1= grid.fit(X1_train, y1_train)
grid_fit2= grid.fit(X2_train, y2_train)
grid1 = timeit.default_timer()
print("gridfit time: ", grid1- gridtime)

grid_score1 = grid_fit1.best_score_
grid_score2 = grid_fit2.best_score_
grid2 = timeit.default_timer()
print("grid score time: ", grid2-grid1)

endzeitPCA = timeit.default_timer()

print("Grid Scores: ", "Score 1: ", grid_score1, "Score 2: ", grid_score2)
print(" ")

print("Zeit für den Principal Component Analysis Algorithmus: ", endzeitPCA-startzeitPCA)
#"""

# In[10]:

"""Neuronal Network with scikit """

#"""
neuralstart = timeit.default_timer()

scaler = StandardScaler()
X1_train_std = scaler.fit_transform(X1_train)
X1_test_std = scaler.fit_transform(X1_test)


neuralloop = timeit.default_timer()

neural_fit = MLPClassifier(activation = "logistic", learning_rate = "adaptive", random_state = 0).fit(X1_train_std, y1_train)

neural_score = neural_fit.score(X1_test_std, y1_test)

neuralend = timeit.default_timer()

print("Time for Neural network: ", neuralend-neuralloop)
print(neural_score)

neuralfinish = timeit.default_timer()

print("The time used is ", neuralfinish-neuralstart)
print("The maximal score is ")
print(np.max(saver))
#"""

# In[11]: KNN


#"""
knnstart = timeit.default_timer()


knnInitialize = timeit.default_timer()

knnGrid_fit1 = knn.fit(X1_train, y1_train)
knnGrid_fit2 = knn.fit(X2_train, y2_train)
knnGrid_PCAfit1 = knn.fit(PCA1_train, y1_train)
knnGrid_PCAfit2 = knn.fit(PCA2_train, y2_train)


knnEnd = timeit.default_timer()
print("knn zeit 2 : ", knnEnd-knnInitialize)

print("Knn 1: ", knnGrid_fit1.score,
      "Knn 2: ", knnGrid_fit2.score)
print("Grid 1 PCA: ", knnGrid_PCAfit1.score,
      "Grid 2 PCA: ", knnGrid_PCAfit2.score)


print("knn total time: ", knnEnd-knnstart)
#"""

