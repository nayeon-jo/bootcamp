Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> emp=pd.read_csv("D:\pandas_jny\data\employee_data.csv")
>>> df.head()
Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    df.head()
NameError: name 'df' is not defined
>>> emp.head()
   avg_monthly_hrs   department  ...    status  tenure
0              221  engineering  ...      Left     5.0
1              232      support  ...  Employed     2.0
2              184        sales  ...  Employed     3.0
3              206        sales  ...  Employed     2.0
4              249        sales  ...  Employed     3.0

[5 rows x 10 columns]
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>> X, Y = load_iris(return_X_y = True)
>>> X['status']=Y
Traceback (most recent call last):
  File "<pyshell#10>", line 1, in <module>
    X['status']=Y
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
>>> df = pd.DataFrame(X, columns=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'])
Traceback (most recent call last):
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\managers.py", line 1662, in create_block_manager_from_blocks
    make_block(values=blocks[0], placement=slice(0, len(axes[0])))
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\blocks.py", line 2722, in make_block
    return klass(values, ndim=ndim, placement=placement)
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\blocks.py", line 130, in __init__
    raise ValueError(
ValueError: Wrong number of items passed 4, placement implies 8

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    df = pd.DataFrame(X, columns=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'])
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 497, in __init__
    mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\construction.py", line 234, in init_ndarray
    return create_block_manager_from_blocks(block_values, [columns, index])
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\managers.py", line 1672, in create_block_manager_from_blocks
    raise construction_error(tot_items, blocks[0].shape[1:], axes, e)
ValueError: Shape of passed values is (150, 4), indices imply (150, 8)
>>> X = emp.drop(labels= ['ststus'], axis = 1)
Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    X = emp.drop(labels= ['ststus'], axis = 1)
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 4163, in drop
    return super().drop(
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 3887, in drop
    obj = obj._drop_axis(labels, axis, level=level, errors=errors)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 3921, in _drop_axis
    new_axis = axis.drop(labels, errors=errors)
  File "D:\anaconda3\lib\site-packages\pandas\core\indexes\base.py", line 5282, in drop
    raise KeyError(f"{labels[mask]} not found in axis")
KeyError: "['ststus'] not found in axis"
>>> emp.head()
   avg_monthly_hrs   department  ...    status  tenure
0              221  engineering  ...      Left     5.0
1              232      support  ...  Employed     2.0
2              184        sales  ...  Employed     3.0
3              206        sales  ...  Employed     2.0
4              249        sales  ...  Employed     3.0

[5 rows x 10 columns]
>>> X = emp.drop(labels= ['status'], axis = 1)
>>> y = emp.status
>>> X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=87, test_size=0.2)
>>> train_rate = round(y_train.sum() / len(y_train),2)
test_rate = round(y_test.sum() / len(y_test),2)
print(f'학습 데이터에서의 Target 비율 : {train_rate}')
print(f'테스트 데이터에서의 Target 비율 : {test_rate}')
SyntaxError: multiple statements found while compiling a single statement
>>> train_rate = round(y_train.sum() / len(y_train),2)
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    train_rate = round(y_train.sum() / len(y_train),2)
TypeError: unsupported operand type(s) for /: 'str' and 'int'
>>> print(train_X[:5],'\n')
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    print(train_X[:5],'\n')
NameError: name 'train_X' is not defined
>>> print(X_train[:5],'\n')
      avg_monthly_hrs  department  ...  satisfaction  tenure
6236              255   marketing  ...      0.108469     4.0
9241              145     finance  ...      0.540461     3.0
9104              259  management  ...      0.942034    10.0
825               144       sales  ...      0.878626     3.0
5884              250         NaN  ...      1.000000     3.0

[5 rows x 9 columns] 

>>> print(X_test[:5],'\n')
       avg_monthly_hrs   department  ...  satisfaction  tenure
8000                55         temp  ...           NaN     NaN
11974              160        sales  ...      0.497947     3.0
13151              148  procurement  ...      0.417884     3.0
7595               189        sales  ...      0.671635     4.0
3631               268        sales  ...      0.759356     3.0

[5 rows x 9 columns] 

>>> print(y_train[:5],'\n')
6236        Left
9241    Employed
9104    Employed
825     Employed
5884    Employed
Name: status, dtype: object 

>>> print(y_test[:5],'\n')
8000     Employed
11974        Left
13151        Left
7595     Employed
3631     Employed
Name: status, dtype: object 

>>> from sklearn import tree
>>> from matplotlib import pyplot as plt
>>> DTmodel = DecisionTreeClassifier()
>>> DTmodel.fit(train_X, train_Y)
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    DTmodel.fit(train_X, train_Y)
NameError: name 'train_X' is not defined
>>> DTmodel.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#28>", line 1, in <module>
    DTmodel.fit(X_train, y_train)
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 890, in fit
    super().fit(
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 156, in fit
    X, y = self._validate_data(X, y,
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 429, in _validate_data
    X = check_array(X, **check_X_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 598, in check_array
    array = np.asarray(array, order=order, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 1781, in __array__
    return np.asarray(self._values, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: could not convert string to float: 'marketing'
>>> pip install scikit-learn
SyntaxError: invalid syntax
>>> DTmodel = DecisionTreeClassifier()
>>> DTmodel.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#31>", line 1, in <module>
    DTmodel.fit(X_train, y_train)
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 890, in fit
    super().fit(
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 156, in fit
    X, y = self._validate_data(X, y,
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 429, in _validate_data
    X = check_array(X, **check_X_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 598, in check_array
    array = np.asarray(array, order=order, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 1781, in __array__
    return np.asarray(self._values, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: could not convert string to float: 'marketing'
>>> 
================================== RESTART: D:/anaconda3/Scripts/20210828.py ==================================
>>> from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
SyntaxError: multiple statements found while compiling a single statement
>>> from sklearn import tree
>>> X = [[0, 0], [1, 1]]
>>> Y = [0, 1]
>>> clf = tree.DecisionTreeClassifier()
>>> clf = clf.fit(X, Y)
>>> clf.predict([[2., 2.]])
array([1])
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.model_selection import train_test_split
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
>>> emp['status']=Y
Traceback (most recent call last):
  File "<pyshell#47>", line 1, in <module>
    emp['status']=Y
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3040, in __setitem__
    self._set_item(key, value)
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3116, in _set_item
    value = self._sanitize_column(key, value)
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3764, in _sanitize_column
    value = sanitize_index(value, self.index)
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\construction.py", line 747, in sanitize_index
    raise ValueError(
ValueError: Length of values (2) does not match length of index (14249)
>>> X, Y = load_iris(return_X_y = True)
>>> emp['status']=Y
Traceback (most recent call last):
  File "<pyshell#49>", line 1, in <module>
    emp['status']=Y
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3040, in __setitem__
    self._set_item(key, value)
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3116, in _set_item
    value = self._sanitize_column(key, value)
  File "D:\anaconda3\lib\site-packages\pandas\core\frame.py", line 3764, in _sanitize_column
    value = sanitize_index(value, self.index)
  File "D:\anaconda3\lib\site-packages\pandas\core\internals\construction.py", line 747, in sanitize_index
    raise ValueError(
ValueError: Length of values (150) does not match length of index (14249)
>>>  X = emp.drop(columns= ['status'])
 
SyntaxError: unexpected indent
>>> X = emp.drop(columns= ['status'])
>>> Y = emp['status']
>>> train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)
>>> emp.head()
   avg_monthly_hrs   department  ...    status  tenure
0              221  engineering  ...      Left     5.0
1              232      support  ...  Employed     2.0
2              184        sales  ...  Employed     3.0
3              206        sales  ...  Employed     2.0
4              249        sales  ...  Employed     3.0

[5 rows x 10 columns]
>>> train_X[:5]
       avg_monthly_hrs department  ...  satisfaction  tenure
409                139    support  ...      0.198545     4.0
1285               250      sales  ...      0.560352     5.0
5030               234    support  ...      0.559908     3.0
13767              202         IT  ...      0.800087     3.0
5763               238    support  ...      0.587424     2.0

[5 rows x 9 columns]
>>> train_Y[:5]
409          Left
1285     Employed
5030     Employed
13767    Employed
5763     Employed
Name: status, dtype: object
>>> test_X
       avg_monthly_hrs   department  ...  satisfaction  tenure
1053               159        sales  ...      0.668489     2.0
487                141        sales  ...      0.854734     3.0
9619               160        sales  ...      0.776747     3.0
927                152      support  ...      0.197001     2.0
2714               143        sales  ...      0.596484     2.0
...                ...          ...  ...           ...     ...
2858               222        sales  ...      0.462791     3.0
13940              176        sales  ...      0.943919     4.0
10661              198  engineering  ...      0.614700     3.0
9606               158  procurement  ...      0.945915     2.0
7995               213        sales  ...      0.731735     5.0

[2850 rows x 9 columns]
>>> test_Y
1053     Employed
487      Employed
9619     Employed
927          Left
2714     Employed
           ...   
2858     Employed
13940    Employed
10661    Employed
9606     Employed
7995     Employed
Name: status, Length: 2850, dtype: object
>>> from matplotlib import pyplot as plt
>>> from sklearn import tree
>>> DTmodel = DecisionTreeClassifier()
>>> DTmodel.fit(train_X, train_Y)
Traceback (most recent call last):
  File "<pyshell#62>", line 1, in <module>
    DTmodel.fit(train_X, train_Y)
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 890, in fit
    super().fit(
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 156, in fit
    X, y = self._validate_data(X, y,
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 429, in _validate_data
    X = check_array(X, **check_X_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 598, in check_array
    array = np.asarray(array, order=order, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 1781, in __array__
    return np.asarray(self._values, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: could not convert string to float: 'support'
>>> fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DTmodel, 
                   feature_names=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'],  
                   class_names=['setosa', 'versicolor', 'virginica'],
                   filled=True)
SyntaxError: multiple statements found while compiling a single statement
>>> fig = plt.figure(figsize=(25,20))
>>> _ = tree.plot_tree(DTmodel, 
                   feature_names=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'],  
                   class_names=['setosa', 'versicolor', 'virginica'],
                   filled=True)
Traceback (most recent call last):
  File "<pyshell#65>", line 1, in <module>
    _ = tree.plot_tree(DTmodel,
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_export.py", line 180, in plot_tree
    check_is_fitted(decision_tree)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 1019, in check_is_fitted
    raise NotFittedError(msg % {'name': type(estimator).__name__})
sklearn.exceptions.NotFittedError: This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
>>> from sklearn.linear_model import LinearRegression
>>> lrmodel=LinearRegression()
>>> lrmodel.fit(train_X, train_Y)
Traceback (most recent call last):
  File "<pyshell#68>", line 1, in <module>
    lrmodel.fit(train_X, train_Y)
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 505, in fit
    X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'],
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 432, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 795, in check_X_y
    X = check_array(X, accept_sparse=accept_sparse,
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 598, in check_array
    array = np.asarray(array, order=order, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 1781, in __array__
    return np.asarray(self._values, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: could not convert string to float: 'support'
>>> pred_X=lrmodel.predict(X)
Traceback (most recent call last):
  File "<pyshell#69>", line 1, in <module>
    pred_X=lrmodel.predict(X)
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 236, in predict
    return self._decision_function(X)
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 216, in _decision_function
    check_is_fitted(self)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 1019, in check_is_fitted
    raise NotFittedError(msg % {'name': type(estimator).__name__})
sklearn.exceptions.NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
>>> B0=lrmodel.intercept_
Traceback (most recent call last):
  File "<pyshell#70>", line 1, in <module>
    B0=lrmodel.intercept_
AttributeError: 'LinearRegression' object has no attribute 'intercept_'
>>> Bi=lrmodel.coef_[i-1]
Traceback (most recent call last):
  File "<pyshell#71>", line 1, in <module>
    Bi=lrmodel.coef_[i-1]
AttributeError: 'LinearRegression' object has no attribute 'coef_'
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn import datasets
>>> sc = StandardScaler()
>>> sc.fit(X_train)
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    sc.fit(X_train)
NameError: name 'X_train' is not defined
>>> sc.fit(train_X)
Traceback (most recent call last):
  File "<pyshell#76>", line 1, in <module>
    sc.fit(train_X)
  File "D:\anaconda3\lib\site-packages\sklearn\preprocessing\_data.py", line 667, in fit
    return self.partial_fit(X, y)
  File "D:\anaconda3\lib\site-packages\sklearn\preprocessing\_data.py", line 696, in partial_fit
    X = self._validate_data(X, accept_sparse=('csr', 'csc'),
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 420, in _validate_data
    X = check_array(X, **check_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 598, in check_array
    array = np.asarray(array, order=order, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
  File "D:\anaconda3\lib\site-packages\pandas\core\generic.py", line 1781, in __array__
    return np.asarray(self._values, dtype=dtype)
  File "D:\anaconda3\lib\site-packages\numpy\core\_asarray.py", line 83, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: could not convert string to float: 'support'
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
>>> em=emp.replace({'Left':0, 'Employed':1})
>>> em.head()
   avg_monthly_hrs   department  filed_complaint  ...  satisfaction  status  tenure
0              221  engineering              NaN  ...      0.829896       0     5.0
1              232      support              NaN  ...      0.834544       1     2.0
2              184        sales              NaN  ...      0.834988       1     3.0
3              206        sales              NaN  ...      0.424764       1     2.0
4              249        sales              NaN  ...      0.779043       1     3.0

[5 rows x 10 columns]
>>> X, Y = load_iris(return_X_y = True)
>>> em=em.drop(columns=['department'])
>>> em.head()
   avg_monthly_hrs  filed_complaint  ...  status  tenure
0              221              NaN  ...       0     5.0
1              232              NaN  ...       1     2.0
2              184              NaN  ...       1     3.0
3              206              NaN  ...       1     2.0
4              249              NaN  ...       1     3.0

[5 rows x 9 columns]
>>> em.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    2058 non-null   float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  300 non-null    float64
 5   salary             14249 non-null  object 
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(3), object(1)
memory usage: 1002.0+ KB
>>> em=em.replace({'low':1, 'medium':2,'high':3})
>>> em.head()
   avg_monthly_hrs  filed_complaint  ...  status  tenure
0              221              NaN  ...       0     5.0
1              232              NaN  ...       1     2.0
2              184              NaN  ...       1     3.0
3              206              NaN  ...       1     2.0
4              249              NaN  ...       1     3.0

[5 rows x 9 columns]
>>> em.head()
   avg_monthly_hrs  filed_complaint  ...  status  tenure
0              221              NaN  ...       0     5.0
1              232              NaN  ...       1     2.0
2              184              NaN  ...       1     3.0
3              206              NaN  ...       1     2.0
4              249              NaN  ...       1     3.0

[5 rows x 9 columns]
>>> em.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    2058 non-null   float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  300 non-null    float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> X, Y = load_iris(return_X_y = True)
>>> X = em.drop(columns= ['status'])
>>> Y = em['status']
>>> train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)
>>> train_X[:5]
       avg_monthly_hrs  filed_complaint  ...  satisfaction  tenure
409                139              NaN  ...      0.198545     4.0
1285               250              NaN  ...      0.560352     5.0
5030               234              NaN  ...      0.559908     3.0
13767              202              NaN  ...      0.800087     3.0
5763               238              NaN  ...      0.587424     2.0

[5 rows x 8 columns]
>>> train_Y[:5]
409      0
1285     1
5030     1
13767    1
5763     1
Name: status, dtype: int64
>>> test_X
       avg_monthly_hrs  filed_complaint  ...  satisfaction  tenure
1053               159              NaN  ...      0.668489     2.0
487                141              NaN  ...      0.854734     3.0
9619               160              NaN  ...      0.776747     3.0
927                152              NaN  ...      0.197001     2.0
2714               143              NaN  ...      0.596484     2.0
...                ...              ...  ...           ...     ...
2858               222              NaN  ...      0.462791     3.0
13940              176              NaN  ...      0.943919     4.0
10661              198              1.0  ...      0.614700     3.0
9606               158              NaN  ...      0.945915     2.0
7995               213              1.0  ...      0.731735     5.0

[2850 rows x 8 columns]
>>> test_Y
1053     1
487      1
9619     1
927      0
2714     1
        ..
2858     1
13940    1
10661    1
9606     1
7995     1
Name: status, Length: 2850, dtype: int64
>>> DTmodel.fit(train_X, train_Y)
Traceback (most recent call last):
  File "<pyshell#96>", line 1, in <module>
    DTmodel.fit(train_X, train_Y)
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 890, in fit
    super().fit(
  File "D:\anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 156, in fit
    X, y = self._validate_data(X, y,
  File "D:\anaconda3\lib\site-packages\sklearn\base.py", line 429, in _validate_data
    X = check_array(X, **check_X_params)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
    return f(**kwargs)
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 644, in check_array
    _assert_all_finite(array,
  File "D:\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 96, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
>>> em.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    2058 non-null   float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  300 non-null    float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> empl=em.replace({'filed_complaint': np.nan}, {'filed_complaint': 0})
>>> empl.head()
   avg_monthly_hrs  filed_complaint  ...  status  tenure
0              221              0.0  ...       0     5.0
1              232              0.0  ...       1     2.0
2              184              0.0  ...       1     3.0
3              206              0.0  ...       1     2.0
4              249              0.0  ...       1     3.0

[5 rows x 9 columns]
>>> empl.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    14249 non-null  float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  300 non-null    float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> empl=em.replace({'recently_promoted': np.nan}, {'recently_promoted': 0})
>>> empl.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    2058 non-null   float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  14249 non-null  float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> empl=empl.replace({'recently_promoted': np.nan}, {'recently_promoted': 0})
>>> empl.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    2058 non-null   float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  14249 non-null  float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> empl=empl.replace({'filed_complaint': np.nan}, {'filed_complaint': 0})
>>> empl.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14249 entries, 0 to 14248
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   avg_monthly_hrs    14249 non-null  int64  
 1   filed_complaint    14249 non-null  float64
 2   last_evaluation    12717 non-null  float64
 3   n_projects         14249 non-null  int64  
 4   recently_promoted  14249 non-null  float64
 5   salary             14249 non-null  int64  
 6   satisfaction       14068 non-null  float64
 7   status             14249 non-null  int64  
 8   tenure             14068 non-null  float64
dtypes: float64(5), int64(4)
memory usage: 1002.0 KB
>>> emp2=empl.dropna()
>>> emp2.info()
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
>>> X = emp2.drop(columns= ['status'])
>>> Y = emp2['status']
>>> train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)
>>> train_X[:5]
       avg_monthly_hrs  filed_complaint  ...  satisfaction  tenure
8770               223              1.0  ...      0.476331     3.0
6350               263              0.0  ...      0.111513     4.0
8756               146              0.0  ...      0.646158     3.0
12997              147              0.0  ...      0.919079     3.0
4822               249              0.0  ...      0.179090     3.0

[5 rows x 8 columns]
>>> DTmodel.fit(train_X, train_Y)
DecisionTreeClassifier()
>>> DTmodel = DecisionTreeClassifier()
>>> DTmodel.fit(train_X, train_Y)
DecisionTreeClassifier()
>>> pred_X=DTmodel.predict(test_X)
>>> plt.show()
>>> train_X.shape
(10173, 8)
>>> score = pd.DataFrame({"Model":['DTmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, pred_X)],
                   "Recall": [recall_score(test_Y, pred_X)],
                   "F1score": [f1_score(test_Y, pred_X]})
		     
SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
>>> Score = pd.DataFrame({'Model':[],
                     'Accuracy Score':[],
                     'Recall':[],
                     'F1score':[]})
>>> score = pd.DataFrame({"Model":['DTmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, pred_X)],
                   "Recall": [recall_score(test_Y, pred_X)],
                   "F1score": [f1_score(test_Y, pred_X]})
		     
SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
>>> score = pd.DataFrame({"Model":['DTmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, pred_X)],
                   "Recall": [recall_score(test_Y, pred_X)],
                   "F1score": [f1_score(test_Y, pred_X)]})
Traceback (most recent call last):
  File "<pyshell#123>", line 3, in <module>
    "Recall": [recall_score(test_Y, pred_X)],
NameError: name 'recall_score' is not defined
>>> from sklearn.metrics import accuracy_score,recall_score, f1_score

>>> y_pred = DTmodel.predict(test_X)
>>> score = pd.DataFrame({"Model":['DTmodel'],
                    "Accuracy Score": [accuracy_score(test_Y, y_pred)],
                   "Recall": [recall_score(test_Y, y_pred)],
                   "F1score": [f1_score(test_Y, y_pred)]})
>>> score
     Model  Accuracy Score   Recall   F1score
0  DTmodel         0.96423  0.96987  0.976209
>>> DTmodel.score(train_X,train_Y)
1.0
>>> precision_score(train_Y,y_pred)
Traceback (most recent call last):
  File "<pyshell#129>", line 1, in <module>
    precision_score(train_Y,y_pred)
NameError: name 'precision_score' is not defined
>>> recall_score=(train_Y,y_pred)
>>> recall_score
(8770     1
6350     0
8756     1
12997    1
4822     1
        ..
13407    1
5845     0
6077     1
965      1
8183     0
Name: status, Length: 10173, dtype: int64, array([1, 0, 1, ..., 1, 1, 1], dtype=int64))
>>> confusion_matrix(y_true,y_pred)
Traceback (most recent call last):
  File "<pyshell#132>", line 1, in <module>
    confusion_matrix(y_true,y_pred)
NameError: name 'confusion_matrix' is not defined
>>> fig = plt.figure(figsize=(25,20))
>>> _ = tree.plot_tree(DTmodel, 
                   feature_names=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'],  
                 filled=True)
>>> _ = tree.plot_tree(DTmodel, 
                   feature_names=['salary','tenure','n_projects','avg_monthly_hrs','satisfaction','last_evaluation','filed_complaint','recently_promoted'],  
                  class_names=['employed', 'left'],
                 filled=True)
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, Y = load_iris(return_X_y = True)
>>> classfier=LogisticRegression(multi_class='auto').fit(X,Y)

Warning (from warnings module):
  File "D:\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 762
    n_iter_i = _check_optimize_result(
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
>>> classfier.predict(X)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
>>> classfier.predict_proba(X)
array([[9.81797141e-01, 1.82028445e-02, 1.44269293e-08],
       [9.71725476e-01, 2.82744937e-02, 3.01659208e-08],
       [9.85444223e-01, 1.45557643e-02, 1.23263078e-08],
       [9.76282998e-01, 2.37169623e-02, 3.97229604e-08],
       [9.85381263e-01, 1.46187255e-02, 1.19450737e-08],
       [9.70457205e-01, 2.95427213e-02, 7.35307149e-08],
       [9.86879212e-01, 1.31207678e-02, 1.99800358e-08],
       [9.76395201e-01, 2.36047710e-02, 2.76315897e-08],
       [9.79831319e-01, 2.01686506e-02, 3.06875994e-08],
       [9.69130364e-01, 3.08696046e-02, 3.16498069e-08],
       [9.76506883e-01, 2.34930977e-02, 1.92207309e-08],
       [9.75396301e-01, 2.46036553e-02, 4.38552739e-08],
       [9.74557148e-01, 2.54428301e-02, 2.14857551e-08],
       [9.91967765e-01, 8.03223109e-03, 3.89483945e-09],
       [9.88209458e-01, 1.17905393e-02, 2.80767915e-09],
       [9.86771830e-01, 1.32281572e-02, 1.27909501e-08],
       [9.88115327e-01, 1.18846639e-02, 9.19925673e-09],
       [9.81552606e-01, 1.84473739e-02, 1.96973245e-08],
       [9.56599657e-01, 4.34002746e-02, 6.83951083e-08],
       [9.84113703e-01, 1.58862761e-02, 2.05275655e-08],
       [9.46785202e-01, 5.32147113e-02, 8.63889140e-08],
       [9.81740024e-01, 1.82599430e-02, 3.28783204e-08],
       [9.96005031e-01, 3.99496731e-03, 1.31264141e-09],
       [9.52320721e-01, 4.76790419e-02, 2.37371887e-07],
       [9.51822159e-01, 4.81776344e-02, 2.06208711e-07],
       [9.51568367e-01, 4.84315461e-02, 8.68910029e-08],
       [9.69618654e-01, 3.03812597e-02, 8.64372737e-08],
       [9.74932791e-01, 2.50671842e-02, 2.49489234e-08],
       [9.77354484e-01, 2.26454988e-02, 1.74089211e-08],
       [9.71213447e-01, 2.87864947e-02, 5.85364149e-08],
       [9.64281670e-01, 3.57182600e-02, 7.04503169e-08],
       [9.64968897e-01, 3.50310450e-02, 5.75611232e-08],
       [9.88374451e-01, 1.16255419e-02, 7.01423788e-09],
       [9.89066890e-01, 1.09331047e-02, 5.29144642e-09],
       [9.68721084e-01, 3.12788727e-02, 4.32045151e-08],
       [9.84673578e-01, 1.53264138e-02, 7.99320451e-09],
       [9.78970817e-01, 2.10291735e-02, 9.62581990e-09],
       [9.86863806e-01, 1.31361860e-02, 8.49595234e-09],
       [9.85845693e-01, 1.41542917e-02, 1.55490650e-08],
       [9.74107662e-01, 2.58923091e-02, 2.84233565e-08],
       [9.86629062e-01, 1.33709263e-02, 1.13690080e-08],
       [9.62258163e-01, 3.77417704e-02, 6.63395770e-08],
       [9.89009064e-01, 1.09909247e-02, 1.12859139e-08],
       [9.72453887e-01, 2.75459759e-02, 1.37522170e-07],
       [9.60230242e-01, 3.97695358e-02, 2.22615994e-07],
       [9.73874260e-01, 2.61257004e-02, 4.00432425e-08],
       [9.80309318e-01, 1.96906564e-02, 2.52901033e-08],
       [9.83337501e-01, 1.66624788e-02, 2.01489408e-08],
       [9.78587115e-01, 2.14128661e-02, 1.86812492e-08],
       [9.78678704e-01, 2.13212765e-02, 1.92780449e-08],
       [2.11818579e-03, 8.74132410e-01, 1.23749404e-01],
       [5.77632428e-03, 8.59770984e-01, 1.34452691e-01],
       [1.05424918e-03, 7.25337760e-01, 2.73607990e-01],
       [1.53489830e-02, 9.39251439e-01, 4.53995783e-02],
       [2.36420294e-03, 8.14553887e-01, 1.83081910e-01],
       [6.90979037e-03, 8.60105917e-01, 1.32984293e-01],
       [3.73293734e-03, 7.17002264e-01, 2.79264798e-01],
       [1.47459316e-01, 8.49449295e-01, 3.09138902e-03],
       [2.76432842e-03, 8.96571249e-01, 1.00664423e-01],
       [4.11355502e-02, 9.11884563e-01, 4.69798865e-02],
       [5.57521781e-02, 9.37679977e-01, 6.56784437e-03],
       [1.50665324e-02, 8.98608025e-01, 8.63254423e-02],
       [9.10068240e-03, 9.76420207e-01, 1.44791108e-02],
       [3.01956806e-03, 7.79325407e-01, 2.17655025e-01],
       [7.42443912e-02, 9.15219018e-01, 1.05365906e-02],
       [5.26660134e-03, 9.26245479e-01, 6.84879200e-02],
       [8.60351161e-03, 7.74702322e-01, 2.16694167e-01],
       [1.63582646e-02, 9.65231114e-01, 1.84106217e-02],
       [1.80303533e-03, 7.99065346e-01, 1.99131618e-01],
       [2.38771632e-02, 9.59397905e-01, 1.67249323e-02],
       [2.27394084e-03, 4.40294838e-01, 5.57431222e-01],
       [1.67889526e-02, 9.56644958e-01, 2.65660893e-02],
       [7.08629071e-04, 5.95132517e-01, 4.04158854e-01],
       [3.01270781e-03, 8.60095538e-01, 1.36891754e-01],
       [7.05154237e-03, 9.42820076e-01, 5.01283814e-02],
       [5.05964635e-03, 9.19917261e-01, 7.50230930e-02],
       [1.11480179e-03, 8.01208733e-01, 1.97676466e-01],
       [5.73551906e-04, 4.80887717e-01, 5.18538731e-01],
       [5.43376164e-03, 8.12720066e-01, 1.81846172e-01],
       [6.17120158e-02, 9.34877066e-01, 3.41091806e-03],
       [2.90651985e-02, 9.57190921e-01, 1.37438808e-02],
       [3.70890045e-02, 9.55307682e-01, 7.60331391e-03],
       [2.50959579e-02, 9.56439251e-01, 1.84647910e-02],
       [4.43187726e-04, 3.49557878e-01, 6.49998935e-01],
       [1.00828285e-02, 7.51038421e-01, 2.38878750e-01],
       [9.87511115e-03, 7.89174015e-01, 2.00950874e-01],
       [2.25065536e-03, 8.05154561e-01, 1.92594784e-01],
       [2.75854701e-03, 9.12420091e-01, 8.48213617e-02],
       [2.68227580e-02, 9.28668577e-01, 4.45086649e-02],
       [1.98000425e-02, 9.37827909e-01, 4.23720485e-02],
       [8.62858604e-03, 8.97770668e-01, 9.36007458e-02],
       [4.60144674e-03, 8.28365656e-01, 1.67032897e-01],
       [1.75132360e-02, 9.56899533e-01, 2.55872306e-02],
       [1.21587918e-01, 8.75322217e-01, 3.08986476e-03],
       [1.43508221e-02, 9.20292035e-01, 6.53571426e-02],
       [1.98090018e-02, 9.38274287e-01, 4.19167111e-02],
       [1.69571030e-02, 9.25432983e-01, 5.76099137e-02],
       [8.45383694e-03, 9.35016409e-01, 5.65297538e-02],
       [2.43882948e-01, 7.54806692e-01, 1.31036059e-03],
       [1.90312396e-02, 9.35964331e-01, 4.50044291e-02],
       [8.85058602e-07, 3.92899392e-03, 9.96070121e-01],
       [2.38414105e-04, 1.62029616e-01, 8.37731970e-01],
       [2.45058631e-06, 2.56169206e-02, 9.74380629e-01],
       [3.07861395e-05, 8.19475307e-02, 9.18021683e-01],
       [3.67208806e-06, 1.74627149e-02, 9.82533613e-01],
       [5.47783053e-08, 4.67282830e-03, 9.95327117e-01],
       [5.66458505e-03, 5.11940598e-01, 4.82394817e-01],
       [6.16198519e-07, 2.15062129e-02, 9.78493171e-01],
       [5.15458730e-06, 5.32440523e-02, 9.46750793e-01],
       [6.45317395e-07, 5.77757472e-03, 9.94221780e-01],
       [2.98223201e-04, 2.10340038e-01, 7.89361739e-01],
       [7.16578564e-05, 1.36871081e-01, 8.63057261e-01],
       [2.09736812e-05, 6.51730036e-02, 9.34806023e-01],
       [2.24656010e-04, 1.44183588e-01, 8.55591756e-01],
       [6.73939787e-05, 4.31298857e-02, 9.56802720e-01],
       [5.07180003e-05, 5.39151498e-02, 9.46034132e-01],
       [5.49147025e-05, 1.23254357e-01, 8.76690728e-01],
       [8.41464632e-08, 3.62195466e-03, 9.96377961e-01],
       [3.10604447e-09, 1.00262175e-03, 9.98997375e-01],
       [3.84703385e-04, 4.50239177e-01, 5.49376120e-01],
       [5.51796910e-06, 2.38545452e-02, 9.76139937e-01],
       [6.01697723e-04, 1.89624708e-01, 8.09773594e-01],
       [3.10785379e-08, 4.68583454e-03, 9.95314134e-01],
       [5.78335011e-04, 3.91597028e-01, 6.07824637e-01],
       [1.26655233e-05, 3.87798490e-02, 9.61207486e-01],
       [4.81993758e-06, 5.19225547e-02, 9.48072625e-01],
       [1.05944841e-03, 4.55096733e-01, 5.43843818e-01],
       [1.00926397e-03, 3.84961664e-01, 6.14029072e-01],
       [1.04679150e-05, 3.62713346e-02, 9.63718197e-01],
       [1.67750701e-05, 1.42813692e-01, 8.57169532e-01],
       [1.05346960e-06, 2.92774764e-02, 9.70721470e-01],
       [7.01252455e-07, 1.76963131e-02, 9.82302986e-01],
       [7.73746804e-06, 2.71772613e-02, 9.72815001e-01],
       [5.22966277e-04, 4.75825778e-01, 5.23651256e-01],
       [6.18542304e-05, 1.89419491e-01, 8.10518655e-01],
       [3.88801379e-07, 1.17375402e-02, 9.88262071e-01],
       [1.14087133e-05, 1.73670104e-02, 9.82621581e-01],
       [6.68131533e-05, 1.19995050e-01, 8.79938137e-01],
       [1.59899983e-03, 4.39792434e-01, 5.58608566e-01],
       [3.91741780e-05, 9.33547256e-02, 9.06606100e-01],
       [6.19410774e-06, 2.02345559e-02, 9.79759250e-01],
       [9.82423605e-05, 1.19886704e-01, 8.80015053e-01],
       [2.38414105e-04, 1.62029616e-01, 8.37731970e-01],
       [2.01727777e-06, 1.26187208e-02, 9.87379262e-01],
       [3.74241295e-06, 1.20989750e-02, 9.87897283e-01],
       [5.50067247e-05, 7.96076880e-02, 9.20337305e-01],
       [2.23703921e-04, 2.50210723e-01, 7.49565573e-01],
       [1.36434669e-04, 1.56810163e-01, 8.43053403e-01],
       [4.48418890e-05, 3.84974295e-02, 9.61457729e-01],
       [4.68256902e-04, 2.35052560e-01, 7.64479184e-01]])
>>> classfier.score(X,Y)
0.9733333333333334
>>> 