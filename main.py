import pandas as pd 
import numpy as np
import plotly.express as px
import nbformat
import xgboost as xgb 

n = 1000

gender = np.random.binomial(1, 0.5, n)
age_group = np.random.binomial(1, 0.5, n)
education_level = np.random.binomial(1, 0.5, n)
knowledge_dutch = np.random.binomial(1, 0.5, n)
nationality = np.random.binomial(1,0.5,n)


data = pd.DataFrame(
    {
        'intercept':1,
        'gender': gender,
        'age_group':age_group,
        'education':education_level,
        'knowledge_dutch':knowledge_dutch,
        'nationality':nationality
    }
)

# distribution of 
counts = data.groupby(['gender',
                      'age_group',
                      'education',
                      'knowledge_dutch',
                      'nationality']).count()


px.histogram(counts,x='intercept')




beta = np.reshape([10, 40, 20, 10, 4,3],(6,1))
beta.shape
X =  np.reshape(np.array(data),data.shape)
X.shape
activity_score = np.reshape(np.matmul(X,beta),(n,)) + np.random.normal(0,10,n)

activity_score

data['activity_score'] = activity_score

means = data.groupby(['gender',
                      'age_group',
                      'education',
                      'knowledge_dutch',
                      'nationality'])['activity_score'].mean()
def str_merge(x):
     x = x.values
     x = [ str(x) for i in x]
     return(x)

data[['gender', 
     'age_group',
     'education',
     'knowledge_dutch',
     'nationality']] = data[['gender', 
                             'age_group',
                             'education',
                             'knowledge_dutch',
                             'nationality']].astype(str)

minority_group =data[['gender','age_group','education',
                      'knowledge_dutch', 'nationality']].agg(lambda x : ''.join(x),axis=1)

data['minority_group'] = minority_group

data.minority_group.unique()
px.histogram(data,x='activity_score',color='minority_group',nbins=200)
px.box(data,y='activity_score',color='minority_group')


index_val = list(data.groupby('minority_group')['activity_score']
                     .mean()
                     .sort_values()
                     .index
                     .values
                     )
px.box(data,
      y='activity_score',
      color='minority_group',
      category_orders={'minority_group':index_val})



cluster = data.groupby('minority_group').agg(Q1=('activity_score',lambda x: np.quantile(x,0.25)),
                                   Q2=('activity_score',lambda x: np.quantile(x,0.50)),
                                   Q3=('activity_score',lambda x: np.quantile(x,0.75)),
                                   Q4=('activity_score',lambda x: np.quantile(x,1)))

data.set_index('minority_group').join(cluster,how='left')

data.drop('intercept',axis=1)

