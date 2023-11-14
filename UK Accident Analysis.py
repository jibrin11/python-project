#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('UK_Accident.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


# Basic data exploration
print("Summary statistics:")
print(df.describe())


# In[ ]:





# In[9]:


#Histogram of Accident Severity
plt.plot(2, 2, 1)
sns.histplot(df['Accident_Severity'], bins=3, kde=False)
plt.title('Accident Severity Histogram')


# In[8]:


Countplot of Day of the Week
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Day_of_Week')
plt.title('Accidents by Day of the Week')


# In[9]:


Boxplot of Number of Vehicles
plt.subplot(2, 2, 3)
sns.boxplot(data=df, y='Number_of_Vehicles')
plt.title('Number of Vehicles Involved in Accidents')


# In[38]:


Road Type and Accident Severity
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Road_Type', hue='Accident_Severity')
plt.xticks(rotation=45)
plt.title('Road Type vs. Accident Severity')

plt.tight_layout()


# In[39]:


#Correlation heatmap (optional)
plt.figure(figsize=(10, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Show the plots
plt.show()


# In[40]:


# What is the overall trend in the number of accidents over the years?
accidents_by_year = df.groupby('Year')['Accident_Index'].count()
plt.plot(accidents_by_year)
plt.title('Number of Accidents Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.show()


# In[41]:


# Are there specific months or days of the week with a higher number of accidents?
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
accidents_by_month = df.groupby('Month')['Accident_Index'].count()
plt.bar(accidents_by_month.index, accidents_by_month)
plt.title('Number of Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.show()


# In[42]:


# What is the distribution of accident severity in the dataset?
sns.countplot(data=df, x='Accident_Severity')
plt.title('Accident Severity Distribution')
plt.show()


# In[43]:


# Can we identify factors that contribute to more severe accidents?
sns.countplot(data=df, x='Weather_Conditions', hue='Accident_Severity')
plt.title('Weather Conditions vs. Accident Severity')
plt.xticks(rotation=45)
plt.show()


# In[44]:


# How does the distribution of accidents vary between urban and rural areas?
sns.countplot(data=df, x='Urban_or_Rural_Area')
plt.title('Distribution of Accidents in Urban and Rural Areas')
plt.show()


# In[45]:


# What time of day sees the highest number of accidents?
accidents_by_time = df.groupby('Time')['Accident_Index'].count().sort_values(ascending=False).head(10)
plt.bar(accidents_by_time.index, accidents_by_time)
plt.title('Top 10 Time Slots with the Highest Number of Accidents')
plt.xlabel('Time')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()


# In[46]:


# How often do police officers attend the scene of an accident?
sns.countplot(data=df, x='Did_Police_Officer_Attend_Scene_of_Accident')
plt.title('Police Attendance at the Scene of an Accident')
plt.show()


# In[47]:


# Question 10: Is there a difference in reporting or severity when police officers are present?
sns.countplot(data=df, x='Did_Police_Officer_Attend_Scene_of_Accident', hue='Accident_Severity')
plt.title('Police Attendance vs. Accident Severity')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




