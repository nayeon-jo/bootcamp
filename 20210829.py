Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> import pandas as pd
>>> from sklearn.metrics import accuracy_score


>>> emp=pd.read_csv("D:\pandas_jny\data\employee_data.csv")
>>> emp.head()
   avg_monthly_hrs   department  ...    status  tenure
0              221  engineering  ...      Left     5.0
1              232      support  ...  Employed     2.0
2              184        sales  ...  Employed     3.0
3              206        sales  ...  Employed     2.0
4              249        sales  ...  Employed     3.0

[5 rows x 10 columns]
>>> emp.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 10 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   department         13540 non-null  object 
 2   filed_complaint    2058 non-null   float64
 3   last_evaluation    12717 non-null  float64
 4   n_projects         14249 non-null  int64  
 5   recently_promoted  300 non-null    float64
 6   salary             14249 non-null  object 
 7   satisfaction       14068 non-null  float64
 8   status             14249 non-null  object 
 9   tenure             14068 non-null  float64
dtypes: float64(5), int64(2), object(3)
memory usage: 1.1+ MB
>>> em=emp.drop(columns=['department'])
>>> em=em.replace({'Left':0, 'Employed':1})
>>> em=em.replace({'low':1, 'medium':2,'high':3})
>>> em=em.replace({'filed_complaint': np.nan}, {'filed_complaint': 0})
>>> em=em.replace({'recently_promoted': np.nan}, {'recently_promoted': 0})
>>> em=em.dropna()
>>> em.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 12717 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    12717 non-null  int64  
 1   filed_complaint    12717 non-null  float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         12717 non-null  int64  
 4   recently_promoted  12717 non-null  float64
 5   salary             12717 non-null  int64  
 6   satisfaction       12717 non-null  float64
 7   status             12717 non-null  int64  
 8   tenure             12717 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 993.5 KB
	
	
	
>>> X, Y = load_iris(return_X_y = True)

>>> X = em.drop(columns= ['status'])
>>> Y = em['status']
>>> train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)



>>> from sklearn.preprocessing import StandardScaler
>>> sc = StandardScaler()
>>> sc.fit(train_X)
StandardScaler()
>>> X_train_std = sc.transform(train_X)
>>> X_test_std = sc.transform(test_X)



>>> from sklearn.svm import SVC
>>> svm_model = SVC(kernel='rbf', C=8, gamma=0.1)
>>> svm_model.fit(X_train_std, train_Y)
SVC(C=8, gamma=0.1)
>>> y_pred = svm_model.predict(X_test_std)
>>> print("예측된 라벨:", y_pred)
예측된 라벨: [1 0 1 ... 1 1 1]

>>> print("ground-truth 라벨:", test_Y)
ground-truth 라벨: 7026     1
2050     0
4084     1
13812    1
12739    1
        ..
4689     1
3871     1
4603     1
6061     1
2933     1
Name: status, Length: 2544, dtype: int64
>>> print("prediction accuracy: {:.2f}".format(np.mean(y_pred == test_Y)))
prediction accuracy: 0.97
	
	
	
	
	
	
>>> from sklearn.metrics import accuracy_score,recall_score, f1_score
>>> score = pd.DataFrame({"Model":['SVMmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, y_pred)],
                   "Recall": [recall_score(test_Y, y_pred)],
                   "F1score": [f1_score(test_Y, y_pred)]})
>>> score
      Model  Accuracy Score    Recall   F1score
0  SVMmodel         0.96934  0.985455  0.979855




>>> from sklearn.linear_model import LogisticRegression

>>> import statsmodels.api as sm
>>> model=sm.Logit(train_Y,train_X)
>>> results=model.fit(method="newton")
Optimization terminated successfully.
         Current function value: 0.446151
         Iterations 7
>>> results.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 status   No. Observations:                10173
Model:                          Logit   Df Residuals:                    10165
Method:                           MLE   Df Model:                            7
Date:                Sun, 29 Aug 2021   Pseudo R-squ.:                  0.2021
Time:                        16:01:15   Log-Likelihood:                -4538.7
converged:                       True   LL-Null:                       -5688.3
Covariance Type:            nonrobust   LLR p-value:                     0.000
=====================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------
avg_monthly_hrs      -0.0061      0.001    -10.624      0.000      -0.007      -0.005
filed_complaint       1.4850      0.106     14.022      0.000       1.277       1.693
last_evaluation      -1.2124      0.165     -7.362      0.000      -1.535      -0.890
n_projects            0.2581      0.025     10.332      0.000       0.209       0.307
recently_promoted     1.9488      0.362      5.383      0.000       1.239       2.658
salary                0.6200      0.041     14.944      0.000       0.539       0.701
satisfaction          3.6507      0.105     34.644      0.000       3.444       3.857
tenure               -0.2480      0.017    -14.226      0.000      -0.282      -0.214
=====================================================================================
"""
>>> results.params
avg_monthly_hrs     -0.006114
filed_complaint      1.484997
last_evaluation     -1.212417
n_projects           0.258067
recently_promoted    1.948836
salary               0.620003
satisfaction         3.650716
tenure              -0.248035
dtype: float64
>>> np.exp(results.params)
avg_monthly_hrs       0.993905
filed_complaint       4.414951
last_evaluation       0.297477
n_projects            1.294425
recently_promoted     7.020513
salary                1.858935
satisfaction         38.502220
tenure                0.780333
dtype: float64
>>> y_pred=results.predict(test_X)
>>> y_pred
7026     0.884524
2050     0.358182
4084     0.978341
13812    0.967460
12739    0.826201
           ...   
4689     0.862868
3871     0.657886
4603     0.878379
6061     0.596886
2933     0.573521
Length: 2544, dtype: float64
>>> def cut(y,threshold):
	Y=y.copy()
	Y[Y>threshold]=1
	Y[Y<=threshold]=0
	return(Y.astype(int))

>>> pred_Y=cut(y_pred,0.5)
>>> pred_Y
7026     1
2050     0
4084     1
13812    1
12739    1
        ..
4689     1
3871     1
4603     1
6061     1
2933     1
Length: 2544, dtype: int32

		
		
		
>>> from sklearn.metrics import confusion_matrix
>>> cfmat=confusion_matrix(test_Y,pred_Y)
>>> print(cfmat)
[[ 166  453]
 [ 144 1781]]
>>> (cfmat[0,0]+cfmat[1,1])/len(pred_Y)
0.7653301886792453
>>> confusion_matrix의 정확도 계산 결과





>>> score = pd.DataFrame({"Model":['DTmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, pred_Y)],
                   "Recall": [recall_score(test_Y, pred_Y)],
                   "F1score": [f1_score(test_Y, pred_Y)]})
>>> score
     Model  Accuracy Score    Recall   F1score
0  DTmodel         0.76533  0.925195  0.856456




>>> lr = LogisticRegression()
>>> lr.fit(train_X, train_Y)

Warning (from warnings module):
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 762
    n_iter_i = _check_optimize_result(
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
LogisticRegression()
>>> lr.score(train_X, train_Y)
0.7735181362429961
	    
	    
	    
	    
>>> em.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 12717 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    12717 non-null  int64  
 1   filed_complaint    12717 non-null  float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         12717 non-null  int64  
 4   recently_promoted  12717 non-null  float64
 5   salary             12717 non-null  int64  
 6   satisfaction       12717 non-null  float64
 7   status             12717 non-null  int64  
 8   tenure             12717 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 993.5 KB
	    
	    
	    
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn import metrics
>>> import statsmodels.api as sm
	    
	    

>>> log_reg = LogisticRegression()
>>> log_reg.fit(train_X, train_Y)

Warning (from warnings module):
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 762
    n_iter_i = _check_optimize_result(
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
LogisticRegression()
	    

>>> x2 = sm.add_constant(X)

>>> model = sm.OLS(Y, x2)
>>> result = model.fit()
>>> print(result.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 status   R-squared:                       0.207
Model:                            OLS   Adj. R-squared:                  0.206
Method:                 Least Squares   F-statistic:                     413.6
Date:                Sun, 29 Aug 2021   Prob (F-statistic):               0.00
Time:                        16:19:30   Log-Likelihood:                -5866.7
No. Observations:               12717   AIC:                         1.175e+04
Df Residuals:                   12708   BIC:                         1.182e+04
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
const                 0.4433      0.022     20.193      0.000       0.400       0.486
avg_monthly_hrs      -0.0008   7.69e-05     -9.904      0.000      -0.001      -0.001
filed_complaint       0.1662      0.010     17.006      0.000       0.147       0.185
last_evaluation      -0.1265      0.022     -5.784      0.000      -0.169      -0.084
n_projects            0.0273      0.003      8.583      0.000       0.021       0.034
recently_promoted     0.1309      0.024      5.475      0.000       0.084       0.178
salary                0.0977      0.005     18.017      0.000       0.087       0.108
satisfaction          0.6193      0.014     44.716      0.000       0.592       0.646
tenure               -0.0319      0.002    -13.655      0.000      -0.036      -0.027
==============================================================================
Omnibus:                     1176.909   Durbin-Watson:                   1.997
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1381.603
Skew:                          -0.779   Prob(JB):                    9.74e-301
Kurtosis:                       2.573   Cond. No.                     1.56e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.56e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
>>> y_pred
7026     0.884524
2050     0.358182
4084     0.978341
13812    0.967460
12739    0.826201
           ...   
4689     0.862868
3871     0.657886
4603     0.878379
6061     0.596886
2933     0.573521
Length: 2544, dtype: float64

	    
>>> pr=log_reg.predict(test_X)
>>> print(pr)
[1 0 1 ... 1 1 1]
>>> print(list(test_Y))

>>> print('정확도 :', metrics.accuracy_score(test_Y, pr))
정확도 : 0.7775157232704403
>>> score = pd.DataFrame({"Model":['model'],
                    "Accuracy Score": [accuracy_score(test_Y, pr)],
                   "Recall": [recall_score(test_Y, pr)],
                   "F1score": [f1_score(test_Y, pr)]})
>>> score
   Model  Accuracy Score    Recall   F1score
0  model        0.777516  0.923636  0.862688
>>> 
