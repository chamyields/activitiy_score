import pandas as pd 
import numpy as np 

n = 1000 

gender = np.random.binomial(1, 0.2, n)
age_group = np.random.binomial(2, 0.2, n)
education_level = np.random.binomial(6, 0.2, n)
knowledge_dutch = np.random.binomial(3, 0.2, n)
nationality = np.random.binomial(1,0.4,n)


pd.DataFrame(
    {
        'gender': gender,
        'age_group':age_group,
        'education':education_level,
        'knowledge_dutch':knowledge_dutch,
        'nationality':nationality
    }
)


theta = 