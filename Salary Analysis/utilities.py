#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:38:46 2022

@author: nick engelthaler & liam mcelligot
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dfvis = pd.read_csv("dfvis.csv") #Reading in data

'''Company Location: 0 = US BASED COMPANY, 1 = NON-US BASED COMPANY'''

def companyLocation(): #gathers information about company location
    
    val_counts = dfvis["Company_Location"].value_counts() #Getting meaning from data
    in_us = (val_counts[0] / dfvis.shape[0])
    not_in_us = (val_counts[1] / dfvis.shape[0])
    percent_notation = "{:,.2f}%".format(in_us*100)
    print("\nThe percent of company positions located in the US: " + percent_notation)
    percent_notation = "{:,.2f}%".format(not_in_us*100)
    print("The percent of company positions not located in the US " + percent_notation)
   
    #Creating pie chart
    labels = 'In the United States', 'Not in the United States'
    sizes = [in_us, not_in_us]
    colors = ['violet', 'coral']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors = colors, autopct='%1.2f%%', startangle=90, )
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    
    ax1.set_title('Company Position Location')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('CPL.png')
    plt.clf()



'''Company Size: 0 = SMALL MARKET CAP, 1 = AVERAGE OR MEDIUM MARKET CAP,
                 2 = LARGE MARKET CAP'''

def companySize():
    
    val_counts = dfvis["Company_Size"].value_counts()
    small = (val_counts[0] / dfvis.shape[0])
    medium = (val_counts[1] / dfvis.shape[0])
    large = (val_counts[2] / dfvis.shape[0])

    #Initializing values and labels for bar chart
    vals= [small, medium, large]
    labels = ["Small Companies", "Medium/Average Companies", "Large Companies",]
    print('\n')
    for val, label in zip(vals, labels):
        print(f"{label} take up", f'{val*100:.2f}''%', " of all company positions") #Getting meaning from data 
    
    #Creating bar Chart
    plt.figure(figsize = (10,5))
    plt.bar(labels, vals, edgecolor = 'black', color = "royalblue")
    plt.xlabel("Company Size")
    plt.ylabel("Pecent of Positions held ")
    plt.title('Job Positions by Company Size')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("CS.png")
    plt.clf
    
def experienceLevel():
    
    val_counts = dfvis["Experience_Level"].value_counts()
    entry = (val_counts[0] / dfvis.shape[0])
    middle = (val_counts[1] / dfvis.shape[0])
    senior = (val_counts[2] / dfvis.shape[0])
    executive = (val_counts[3] / dfvis.shape[0])
    
    #Initializing values and labels for bar chart
    vals= [entry, middle, senior, executive]
    labels = ["Entry level jobs", "Middle level jobs", "Senior level jobs", "Executive level jobs "]
    print('\n')
    for val, label in zip(vals, labels):
        print(f"{label} take up", f'{val*100:.2f}''%', "of the job market") #Getting meaning from data 
    
    #Creating bar Chart
    plt.figure(figsize = (10,5))
    plt.bar(labels, vals, edgecolor = 'black', color = "m")
    plt.xlabel("Job Positions")
    plt.ylabel("Pecent Held")
    plt.title('Job Positions Held')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("EL.png")
    plt.clf
#Salary stats
def salaryStats():

    '''Summary Statistics'''    
    salary_df = dfvis["Salary_In_USD"] #Reading in only 'Salary_In_USD'
    print('\n')
    s_mean = np.mean(salary_df) #Mean
    print("The mean of all salaries is: ", "${:,.2f}".format(s_mean))
    s_median = np.median(salary_df) #Median
    print("The median of all salaries is: ", "${:,.2f}".format(s_median))
    s_mode = stats.mode(salary_df) #Mode
    print("The mode of all salaries is: ", s_mode)
    s_standev = np.std(salary_df) #Standard Deviation
    print("The standard deviation of all salaries is: ", "{:,.3f}".format(s_standev))
    s_var = np.var(salary_df) #Variance 
    print("The variance of all salaries is: ", s_var)
    
    '''Percentiles and Distribution'''
    print('\n')
    s_25 = np.percentile(salary_df, 25) #25th percentile
    print("The 25th percintile of all earners can expect a salary of : ", "${:,.2f}".format(s_25))
    s_50 = np.percentile(salary_df, 50) #50th percentile
    print("The 50th percintile of all earners can expect a salary of : ", "${:,.2f}".format(s_50))
    s_75 = np.percentile(salary_df, 75) # 75th percentile
    print("The 75th percintile of all earners can expect a salary of : ", "${:,.2f}".format(s_75))
    s_90 = np.percentile(salary_df, 90) #90th percentile
    print("The 90th percintile of all earners can expect a salary of : ", "${:,.2f}".format(s_90))
    s_99 = np.percentile(salary_df, 99) # 99th percentile for top earners 
    print("The top 99th percintile of all earners can expect a salary of : ", "${:,.2f}".format(s_99))
    
    #histogram of salaries
    plt.figure(figsize = (10,5))
    plt.hist(salary_df, edgecolor='black',  color = "mediumslateblue", bins = 40)
    plt.xlabel("Salary by $USD in Thousands")
    plt.ylabel("Number of Jobs Held")
    plt.title('Distribution of Jobs by Salary')
    plt.show()
    fig1 = plt.gcf()
    fig1.savefig("SS.png")
    plt.clf
    
dfml = pd.read_csv('dfml.csv')

def mlModel():
    
    
    '''Lines 154-175 are responsible for turning turning categorical data into
    numeric data'''
    
    dfml['work_year'] = dfml['work_year'].astype('category') # 0-2020, 1-2021, 2-2022
    dfml['work_year'] = dfml['work_year'].cat.codes
    
    dfml.loc[dfml['experience_level'] == 'EN', 'Experience_level'] = '0'
    dfml.loc[dfml['experience_level'] == 'MI', 'Experience_level'] = '1'
    dfml.loc[dfml['experience_level'] == 'SE', 'Experience_level'] = '2'
    dfml.loc[dfml['experience_level'] == 'EX', 'Experience_level'] = '3'
    
    dfml.loc[dfml['employment_type'] == 'FL', 'Employment_type'] = '0'
    dfml.loc[dfml['employment_type'] == 'PT', 'Employment_type'] = '1'
    dfml.loc[dfml['employment_type'] == 'CT', 'Employment_type'] = '2'
    dfml.loc[dfml['employment_type'] == 'FT', 'Employment_type'] = '3'
    
    dfml.loc[dfml['employee_residence'] == 'US', 'Employee_residence'] = '1'
    dfml.loc[dfml['employee_residence'] != 'US', 'Employee_residence'] = '0'
    
    dfml.loc[dfml['company_location'] == 'US', 'Company_location'] = '1'
    dfml.loc[dfml['company_location'] != 'US', 'Company_location'] = '0'
    
    dfml.loc[dfml['company_size'] == 'S', 'Company_size'] = '0'
    dfml.loc[dfml['company_size'] == 'M', 'Company_size'] = '1'
    dfml.loc[dfml['company_size'] == 'L', 'Company_size'] = '2'
    
    '''X is responsible for dropping old column names, as well as unnessecary
    columns with irrelevant data, essentially prepping dependent variables'''
    
    X = dfml.drop(['experience_level', 'employment_type', 'employee_residence', 
                      'company_location', 'company_size', 'job_title', 'salary', 
                      'salary_currency', 'salary_in_usd'], axis = 1)
    
    
    y = dfml['salary_in_usd'] #Prepping 'C' or independent variable

    lr = LinearRegression() #Intitializing regression object

    '''Training Data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)
    lr.fit(X_train, y_train) #Fitting training data
    intercept = lr.intercept_ #Getting Y-intercept
    coeffs = lr.coef_ #Gettimg coefficients 
    y_pred_train = lr.predict(X_train) #Getting training Y-intercepts
    y_pred_r2_train = r2_score(y_train, y_pred_train) #R score of training data
    '''Training Data Plot'''
    plt.figure(figsize = (10,5))
    plt.scatter(y_train, y_pred_train)
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Training Data 80%')
    plt.show()
    fig1 = plt.gcf()
    fig1.savefig("TRD.png")
    plt.clf
    
    '''Test Data/Plot'''
    y_pred_test = lr.predict(X_test)
    y_pred_r2_test = r2_score(y_test, y_pred_test)
    plt.figure(figsize = (10,5))
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Test Data 20%')
    plt.show()   
    fig1 = plt.gcf()
    fig1.savefig("TED.png")
    plt.clf
       

    lr.fit(X.values,y) #Fitting model to X and y df's
    predicted_sal = lr.predict([[2,100,2,3,1,1,1]]) #Custom salary prediction from variables
    percent_rscore = (lr.score(X.values,y)*100) #Coefficient of determination for prediction


    '''Data Summary'''
    print("\nThe intercept for the data set is: " + "{:,.2f}".format(intercept))
    print("The coeffcients for the dependent variables set are: ", coeffs)
    print("The predicted salary for the variables entered is: ", predicted_sal)
    print('The coefficient of determination for the user model prediction is: ', "{:,.2f}%".format(percent_rscore))
