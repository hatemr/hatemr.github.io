---
Title: Notes on Data Exploration
Date: 2019-09-03
Author: Robert Hatem
Lang: en
Tags:
Image: images/life_expectancy.png
mathjax: true
---

I am reviewing some basic topics in data science to regresh my memory. These are my notes.

### `.corr()`
We will demonstrate a set of basic data analysis functions from Pandas using the toy Boston housing dataset. I chose this dataset since it comes from `sklearn` and can be reproduced easily. First, I will demonstrate `.corr()`.


```python
import pandas as pd
from sklearn.datasets import load_boston
%matplotlib inline
```


```python
boston = load_boston()

# create dataframe
df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
df['MEDV'] = boston['target']

# choose only 2 predictor variables plus target
df = df[['CRIM','DIS','MEDV']]

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>DIS</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>4.0900</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>4.9671</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>4.9671</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>6.0622</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>6.0622</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



Features
* __CRIM__ - per capita crime rate by town
* __DIS__ - weighted distances to five Boston employment centres

Target
* __MEDV__ - Median value of owner-occupied homes in \$1000’s

Let's take the correlation with the target variable.


```python
# correlation
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>DIS</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>1.000000</td>
      <td>-0.379670</td>
      <td>-0.388305</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-0.379670</td>
      <td>1.000000</td>
      <td>0.249929</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>-0.388305</td>
      <td>0.249929</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



__Conclutions__:
* Crime has negative correlation with median home value, because higher crime drives down home values.
* Distance-to-jobs has positive correlation with median home value, which might be surprising. Intuitively, people prefer shorter commutes. One explanation is people prefer the space of suburban homes over urban homes, despite the longer commutes.

Next, I will show `scatter_matrix()`

### `scatter_matrix()`
Plots pairwise scatter plots, with univariate histograms on the diagonal.


```python
pd.plotting.scatter_matrix(df);
```


![png](/images/output_8_0.png)


Conclusions:
* Note the top-middle plot of Crime with distance-to-jobs. The high-crime areas seem to be close to job centers.
* In the bottom-left, the highest value houses are located in low-crime areas, which makes sense.

### `.hist()`
This plots histograms or each column.


```python
df.hist();
```


![png](/images/output_11_0.png)


__Observations__:
* For MEDV, there are a small number areas of high-value homes (right tail).
* Distance-to-jobs has many areas that are close to jobs, then gradually trails off to areas that are far-from-jobs. This makes sense when jobs are in dense urban centers, and density slowly drops as you move away.

### `.bar()`
Represent categorical data with vertical bars with lengths proportional to the values they represent.

First, I will bin the Crime variable by quintiles. Then I will count the frequency, then plot using `.bar()`. Since I'm taking quintiles, each bin will have the same frequency count.


```python
print(pd.qcut(df.CRIM, 5).value_counts())

pd.qcut(df.CRIM, 5).value_counts().plot.bar();
```

    (0.00532, 0.0642]    102
    (5.581, 88.976]      101
    (0.55, 5.581]        101
    (0.15, 0.55]         101
    (0.0642, 0.15]       101
    Name: CRIM, dtype: int64



![png](/images/output_14_1.png)