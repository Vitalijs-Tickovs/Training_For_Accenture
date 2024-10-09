#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

# we don't like warnings
# you can comment the following 2 lines if you'd like to
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


data_path = "../mlcourse.ai_Dataset/"
data = pd.read_csv(data_path + 'adult.data.csv')
data.head()


# In[51]:


### Task 1

print(f'The count of male and female are: \n {data.sex.value_counts()}')


# In[52]:


### Task 2

print(f'The average age of woman: {data[data['sex']=='Female']['age'].mean():.2f}')


# In[53]:


### Task 3

print(f'The distribution of Germans citizens: {data['native-country'].value_counts(normalize=True)['Germany']*100:.2f}%')


# In[106]:


### Task 4-5

age_gt_50k = data[data['salary']=='>50K']['age']

print(f'For adults who earn >50k the mean age is: {round(age_gt_50k.mean(),0)} and std: {age_gt_50k.std()}')

age_lte_50k = data[data['salary']=='<=50K']['age']

print(f'For adults who earn <=50k the mean age is: {round(age_lte_50k.mean(),0)} and std: {age_lte_50k.std()}')


# In[55]:


### Task 6

hs_education = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']

unique_education = data[data['salary']==">50K"]['education'].unique().tolist()

print(f'Do people with salary >50K have only these degrees? {len(set(unique_education)-set(hs_education))==0}')


# In[56]:


### Task 7

groupby = data.groupby(by=['race', 'sex'])['age']

print(f'Age statistic for each race and gender:\n {groupby.describe()}')

max_male_eskimo = groupby.get_group(('Amer-Indian-Eskimo', 'Male')).max()

print(f'The maximum age of men of "Amer-Indian-Eskimo": {max_male_eskimo}')


# In[107]:


### Task 8

married = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']

married_men = data[(data['marital-status'].isin(married)) & (data['sex']=='Male')]

non_married_men = data[~(data['marital-status'].isin(married)) & (data['sex']=='Male')]

salary_gt_50K_married_men = married_men['salary'].value_counts(normalize=True)['>50K']

salary_gt_50K_non_married_men = non_married_men["salary"].value_counts(normalize=True)['>50K']

if salary_gt_50K_married_men> salary_gt_50K_non_married_men:
    print(f'Men who are married tend to earn more: {salary_gt_50K_married_men} vs who is not married: {salary_gt_50K_non_married_men}')
else:
    print(f'Men who are not married tend to earn more: {salary_gt_50K_non_married_men} vs who is married: {salary_gt_50K_married_men}')


# In[88]:


### Task 9

max_work_hours = data['hours-per-week'].max()

people_who_work_max_hours = data[data['hours-per-week']==max_work_hours]

distribution_of_salary = people_who_work_max_hours['salary'].value_counts(normalize=True)

print(f'The maximum # of hours of work per week is: {max_work_hours}')

print(f'People who work such # of hours {people_who_work_max_hours.shape[0]}')

print(f'Percentage of those who earn ">50K" {distribution_of_salary['>50K']*100:.2f}%')


# In[101]:


### Task 10

groupby = data.groupby(by=['native-country', 'salary'])['hours-per-week'].mean()

print(f'Count of average work hours based on "native-country" and their "salary":\n {groupby}\n')

print(f'Data for "Japan": \n{groupby.loc[pd.IndexSlice['Japan'],:]}')


# In[ ]:




