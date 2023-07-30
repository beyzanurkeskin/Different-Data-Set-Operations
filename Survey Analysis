#!/usr/bin/env python
# coding: utf-8

# # 

# # YZV211 - HW3 - Explatory Analysis & Statistical Inference
# 
# In this assignment, you will be proceeding with analysing your answers from the survey. **Read each question thoroughly, write the code for visualization and make an explanation if question asks for. Do not forget to read the assignment document!**

# In[1]:


# Name and Surname: Beyza Nur Keskin
# Student ID : 150200320


# # 

# We first start by importing the necessary libraries for our homework.

# In[93]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statistics
import seaborn as sns


# In[3]:


df = pd.read_csv('survey_v2 (2).csv')     # I  read my csv file
display(df.head())                        # I  display my data set


# # 

# ## Q1.1) Just to keep the column names simple, please rename them here
# 
# You may need view the exact column names

# In[4]:


df.columns                       # I viewed my data sets columns


# In[5]:


# I rename the columns of my dataset

df = df.rename(columns={'GPA': 'GPA',                    #I didn't change this column name because it's short and concise.
                        'Repeat_Course_Num': 'Repeat_Course',
                        'Expected_GPA': 'Expected_GPA',  #I didn't change this column name because it's short and concise.
                        'Expected_Letter_Grade': 'Expected_Grade',
                        'Travel_Time': 'Travel_Time',    #I didn't change this column name because it's short and concise.
                        'Do you have to work to cover your living expenses? ': 'Is_Working',
                        'Time_For_Study': 'Study_Time', 
                        'Time_For_Hobbies': 'Hobby_Time',
                        'Time_For_Sports': 'Sport_Time', 
                        'Time_For_Socializing': 'Socializing_Time',
                        'Time_For_Entertainment': 'Entertainment_Time', 
                        'How would you describe your regular diet? ': 'Diet_Rate',
                        'How many sugary drinks (coke, fruit juice, sweet hot drinks etc) do you consume a day ? ': 'Drink_Num', 
                        'Being good at Maths requires ': 'Math_Rate',
                        'Being good at Programming requires ': 'Programming_Rate', 
                        'Being good at Arts/Music requires ': 'Art_Rate',
                        'Being good at Sports requires ': 'Sport_Rate'})


# In[6]:


display(df.head())                       # I viewed my dataset to check the new names of the columns


# # 

# ## Q1.2a) Apply mapping on the expected letter grades from string to floating point number
# 
# You may need to have a look at to our school's website to convert the letter grades from string to floating point number. In this regard, the grades should be appearing as AA --> 4.0, BA --> 3.5, DD --> 1.0, FF --> 0.0.

# In[7]:


## FIRST WRITE A MAPPING FUNCTION, THEN APPLY IT ON THAT SPECIFIED COLUMN

def mapper(x: str) -> float:                                            # 4. I start the function with the letter note I got

    if x == "AA":                                                       # 5. If letter grade is equal to AA
    
        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 4)       # 6. I update the value to 4

    elif x == "BA":                                                # 7. I apply the same operations for other letter grades.

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 3.5)    # 8.  And update the values

    elif x == "BB":

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 3)

    elif x == "CB":

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 2.5)
        
    elif x == "CC":

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 2)

    elif x == "DC":

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 1.5)

    elif x == "DD":

        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 1)

    elif x == "FF":
        
        df['Expected_Grade'] = df['Expected_Grade'].replace(x, 0)

    
grade_list = df["Expected_Grade"].unique().tolist()                       # 1. I create a list from Expected Grades

for i in grade_list:                                                      # 2. I'm getting each element inside this list
    mapper(i)                                                             # 3. I called the function

df.head()                                                                 # 9. I display my data set


# # 

# ## Q1.2b) Apply mapping on the "Is_Working" column to represent it binary
# 
# "Yes" would correspond to a 1 whereas a "No" would correspond to a 0.

# In[8]:


## CREATE A SIMILAR MAPPING FOR YES/NO TYPE OF ANSWERS

def mapper_binary(x: str) -> int:                          # 4. I start the function with the working situation I got
    
    if x == "Yes":                                         # 5. If student is working
        df['Is_Working'] = df['Is_Working'].replace(x,1)   # 6. I update the value to 1
             
    elif x == "No":                                        # 7. If student is not working
        df['Is_Working'] = df['Is_Working'].replace(x,0)   # 8. I update the value to 0
        

work_list = df["Is_Working"].unique().tolist()             # 1. I create a list from working situation

for i in work_list:                                        # 2. I'm getting each element inside this list
    mapper_binary(i)                                       # 3. I called the function
    
df.head()                                                  # 9. I display my data set


# # 

# # Q2) Describe Data

# Try to express the features of the dataset by using 5-figure summary and mean & standard deviation statistics. Describe what you see down below.

# A2)

# In[10]:


## DESCRIBE THE DATASET

#maximum minimum  the lower and upper quartiles the median mean & standard deviation

describe = df.describe()       # I display my data sets information 
 
describe.drop("count").T       # I dropped the count column because I needed to display average, 
                               # standard deviation, minimum and maximum value, first and third quartile
                               # and median (shown with 50% in the table) statistics


# ### My Comment 

# When I checked the gpa, I saw a value as I expected. Honestly, I wasn't expecting much success.
# 
# Half of the class almost certainly seems to have failed one lesson.
# 
# The expected GPA seems realistic, although it is above the current GPA, it is close.
# 
# there is an expectation of students that they will get a high grade from this course expected_grade shows this
# 
# The average travel time for Istanbul seems reasonable.
# 
# Very few students working
# 
# Enough time is allocated for hobbies, socialization and sports.
# 
# Although the number of sugary drinks consumed is not high and the students do not feel obliged to work, the degree of nutrition is quite low.
# 
# Most students have academic and personal skills

# # 

# # Q3) Comment on the Data

# Answer the following questions on the data while providing graphs as a way to support your answers.

# Q3a) What is the ratio between the students who work and those who do not?

# A3a):

# In[233]:


# PROVIDE AN ACCEPTABLE GRAPH TO SUPPORT THIS CLAIM

plt.figure(figsize=(10, 8))                               # I adjusted the size of the graph
 
plt.xlabel("Working Status")                              # I named the x-axis
plt.ylabel("Number of Students")                          # I named the y-axis
plt.title("Working Status Rate")                          # I named the graph


plt.hist("Is_Working",data = df,bins=3,color="#096D52")   # I plotted a histogram plot from the df data based on
                                                          # the Is_Working feature


plt.show()                                                # I plotted the plot


# ### My Comment 

# As I have examined before, the graph also shows that the number of working students is quite low.

# # 

# Q3b) What is the average of the Expected GPA? Also, is there any anomaly on its value distribution?

# A3b):

# In[237]:


# PROVIDE AN ACCEPTABLE GRAPH TO SUPPORT THIS CLAIM

print("The average of the Expected GPA is:",df["Expected_GPA"].mean() )                                # I calculate mean of expected GPA 


# In[206]:


plt.figure(figsize=(20, 8))                               # I adjusted the size of the graph
 
plt.xlabel("Expected GPA")                                # I named the x-axis
plt.ylabel("Number of Students")                          # I named the y-axis
plt.title("Expected GPA Rate")                            # I named the graph


plt.hist("Expected_GPA",data = df,bins=50,color="orange") # I plotted a histogram plot from the df data based on
                                                          # the Expected_GPA feature


plt.show()                                                # I plotted the plot


# ### My Comment 

# It seems to show a normal distribution so I want to make sure by drawing a graph.

#  

# In[239]:


fig = plt.figure(figsize =(10, 5))
    
plt.title(x)
b = sns.distplot(df["Expected_GPA"])
plt.show()


# I don't see an abnormal value i think i am facing an expected distribution

# # 

# Q3c) What is the avergae of the Expected Letter Grade? Also, is there any anomaly on its value distribution?

# A3c):

# In[242]:


# PROVIDE AN ACCEPTABLE GRAPH TO SUPPORT THIS CLAIM

print("The average of the Expected Letter Grade is:",df["Expected_Grade"].mean())                                    # I calculate mean of expected grade


# In[231]:


plt.figure(figsize=(10, 8))                                    # I adjusted the size of the graph
  
plt.xlabel("Grade Status")                                     # I named the x-axis
plt.ylabel("Number of Students")                               # I named the y-axis
plt.title("Grade Rate")                                        # I named the graph

plt.hist("Expected_Grade",data = df,bins=15,color="#46326D")   # I plotted a histogram plot from the df data based on
                                                               # the Expected_Grade feature

plt.show()                                                     # I plotted the plot


# ### My Comment

# While everyone was expecting high grades, I saw one abnormal grade. It also corresponds to DD.

# # 

# Q3d) Is there a strong correlation ($r_{xy} > 0.5 || r_{xy} < -0.5$) between the travel time and Expected Letter Grade? If so, what is the sample Pearson correlation coefficient?

# A3d):

# In[79]:


# To calculate correlation i needed to calculate covariance first, thus

def cov(x,y):                                    # I created a covariance funciton
    
    
    total_x = 0                                  # I created a variable that stores the total for x
    total_y = 0                                  # I created a variable that stores the total for y
    
    
    [total_x := total_x + i for i in x]          # I added up all the numbers in x
    [total_y := total_y + i for i in y]          # I added up all the numbers in y
    
          
    x_mean = total_x / len(x)                    # I found mean for x
    y_mean = total_y / len(y)                    # I found mean for y
     
    
       
    summary = sum((xi - x_mean) * (yi - y_mean) for xi,yi in zip(x,y))  # I performed the operations required to calculate
                                                                        #  the covariance for x and y simultaneously
        
    cov = summary / (len(x) - 1)                 # I calculated covariance
 
    return cov                                   # I return the covariancev


# In[80]:


def corr(x,y):                       # I created a correlation funciton
    
    std_x = statistics.stdev(x)      # I calculated standard deviation for x
    std_y = statistics.stdev(y)      # I calculated standard deviation for y
    r = cov(x, y) / std_x / std_y    # I calculated correlation
    
    return r                         # I return the correlation


# In[82]:


print("Correlation between the travel time and expected grade is:",corr(df["Expected_Grade"],df["Travel_Time"]))


# ### My Comment 

# There is a weak correlation between travel time and expected grade. We will see this more clearly when I draw the corallason matrix below.

# # 

# Q3e) Define a threshold for travel time in which you will be able to group the individuals that are proximate and distant. Then, describe these groups to see which features have changed significantly and remained unchanged. Write the things that you noticed down below.

# A3e):

# In[18]:


# WRITE YOUR IMPLEMENTATION HERE!! 
TAU = 0 # Change this to a sensible value
        # Group the dataframe based on the TAU value
        # Desribe the statistics of these 2 newly-created dataframes


# # 

# Q3f) Define a threshold of $3.0$ for GPA and draw 3 plots side-by-side in which the x-axis is defined as the time for study and y-axis is chosen among time for hobbies, sports and socializing. Select an appropriate plot type for this task. Use green color to mark GPAs greater than or equal to $3.0$ and blue for GPAs lower than $3.0$. Do you notice any difference in preferences among these two segments? Do not forget to use a legend! 

# A3f):

# In[86]:


import matplotlib.patches


# In[87]:


levels, categories = pd.factorize(df['GPA'])
colors = [plt.cm.tab10(i) for i in levels] 
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


# In[243]:


with sns.axes_style('dark'):
    sns.jointplot(data=df, x="Socializing_Time", y="Study_Time", hue="GPA")


# In[95]:


with sns.axes_style(style='ticks'):
    g = sns.factorplot("GPA", "Socializing_Time", "Study_Time", data=df, kind="box")
    g.set_axis_labels("Day", "Total Bill")


# In[91]:


# WRITE YOUR IMPLEMENTATION HERE!! 
plt.figure(figsize=(15,10))

scatter = plt.scatter(df['Study_Time'].values,df['Socializing_Time'].values,c = colors, alpha=0.5)

plt.legend(handles=handles,  title='Color')

plt.show()


# Q3g) Use the same GPA threshold and group the data frame into two segments as in the previous part. Select an appropriate plot type for depicting the time used for studying, sports and social activities, hobbies, and entertainment. Please include a legend for highlighting the differences between these two segments, and write down below if there are any significant differences in the preference of spending time.

# A3g:)

# In[20]:


# WRITE YOUR IMPLEMENTATION HERE!! 


# Q3h) Do the same experimentation as (Q3g) for identifying whether there is any difference between the segments in terms of the requirements for math, programming, arts/music and sports.

# A3h:)

# In[21]:


# WRITE YOUR IMPLEMENTATION HERE!! 


# # 

# # Q4) Covariance/Correlation Matrices 

# Q4a) Find the covariance and correlation matrices of the variables, and plot them on a good-looking heatmap. You can use the pandas/sklearn functions for this task. Comment on your results.

# In[209]:


# WRITE YOUR IMPLEMENTATION HERE!! 

matrix = df.corr()                          # I found corr matrix for my data set
print("Correlation matrix is : ", matrix)   # Then i printed it


# In[171]:


df.corr().style.background_gradient().set_precision(2)   # I plot my corr matrix 


# In[173]:


fig = plt.figure(figsize=(15,15))                      # I plot my corr matrix using heatmap
sns.heatmap(df.corr())
plt.show()


# Q4b) Based on the correlation matrix, which of the variables are strongly ($r_{xy} > 0.5 || r_{xy} < -0.5$) correlated? Which variable tuple has the highest (anti)correlation?

# In[191]:


corr = df.corr() 
raw =corr[(corr.abs()>0.50) & (corr.abs() < 1.0)].any()   #  I check for strong correlated features
print(raw)


# In[192]:


corr_pairs = df.corr().unstack()                          
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
positive_pairs = sorted_pairs[sorted_pairs > 0.5 ]       # I tried to find which of the variables are strongly correlated
print(positive_pairs)


# ### My Comment 

# Unexpectedly, the time spent on socializing was higher among working students. I want to examine what other different values will come out, so I will do a correlation analysis according to GPA.

#  

# In[175]:


df.corr()["GPA"].abs().nlargest(10)    # I was wondering about the variables that correlate with gpa


# In[176]:


df.corr()["GPA"].nlargest(10)         # I was wondering about the variables that correlate with gpa


# Again, as I did not expect, the grades of the students far from the school seem higher. But it's no surprise to me that entertainment and diet are also positively correlated with GPA.

# # 

# # Q5) Hypothesis Testing

# Select an appropriate test name and type for the hypothesis questions below and provide the p-value as well while defining the significance. You may want to use SciPy here.

# Q5a) Do working students study less than their non-working counterparts?

# In[113]:


# WRITE YOUR IMPLEMENTATION HERE!! 


# Since it contains ones and zeros, I wanted to examine the IW column by turning it into a separate data frame.

work = df.groupby(df.Is_Working)
working = work.get_group(1)

non_work = df.groupby(df.Is_Working)
non_working = non_work.get_group(0)


# In[114]:


working.head()       # I reviewed my new dataset of working students


# In[115]:


non_working.head()   # I reviewed my new dataset of non-working students


# In[124]:


#I printed the average study hours according to whether they work or not.

print("")
print("                       Mean of time for studying\n ")
print("Working Students               " , working["Study_Time"].mean())
print("Non-working Students           " , non_working["Study_Time"].mean())


# ### My Comment

# Working students continue to amaze me because I see them taking more time to study.

# # 

# Q5b) Do students having lower GPA tend to believe intelligence is more key for programming?

# In[136]:


# WRITE YOUR IMPLEMENTATION HERE!! 

#I rated GPAs below 3 as low and created new datasets

higher = df[df["GPA"] >= 3 ]
lower = df[df["GPA"] <3]


# In[137]:


lower.head()   # I reviewed my new dataset of students w\lower GPA 


# In[138]:


higher.head()  # I reviewed my new dataset of students w\higher GPA 


# In[160]:


#I compared the ratios of programming abilities according to their gpa status

print("")
print("                                 mean of degree of")
print("                                 programming skills\n")
print("students w\lower GPA            " , lower["Programming_Rate"].mean())
print("students w\higher GPA           " , higher["Programming_Rate"].mean())


# ### My Comment 

# I didn't get any surprising or thought provoking information here.

# # 

# Q5c) Do students who do sports have a better diet score?   (MY QUESTION)

# In[ ]:


plt.figure(figsize=(10, 8))                                    # I adjusted the size of the graph
  
plt.xlabel("Grade Status")                                     # I named the x-axis
plt.ylabel("Number of Students")                               # I named the y-axis
plt.title("Grade Rate")                                        # I named the graph

plt.hist("Expected_Grade",data = df,bins=15,color="#46326D")   # I plotted a histogram plot from the df data based on
                                                               # the Expected_Grade feature

plt.show()                                                     # I plotted the plot


# In[232]:


# WRITE YOUR IMPLEMENTATION HERE!! 

fig = plt.figure(figsize=(7,7))                                             # I adjusted the size of the graph

plt.gca().spines['top'].set_visible(False)                                  # I adjusted the visiblity
plt.gca().spines['right'].set_visible(False)                                # I adjusted the visiblity

plt.scatter(df["Diet_Rate"],df["Sport_Time"],color='#892068', alpha=0.3)    # I plotted a scatter plot

plt.xlabel("diet rate")                                                     # I named the x-axis 
plt.ylabel("time for sport")                                                # I named the y-axis
plt.title("Contribution of sports to diet rate")                            # I named the graph

plt.show()                                                                  # I plotted the plot


# ### My Last Comment 

# In fact, I was expecting a high level of diet for those who do sports, but it turned out to be very average.
