# Multiple-Regresion-ML-Model-Analysis
This is a 5 function project that aims to look at factors contributing to salaries in "Big Data" from a dataset in kaggle, and provide visualization as well as statistics on how certain values/subcategories perform.  The output of each function produces a graphical representation of the data after each calculation, as well a print statement. The final result is a utilities file, as well as a csv containing the data and a main.py file to showcase the modular programming.
# Prerequisites
I used Python 3.9 for this project, on the latest version of spyder in the anaconda navigator at the time. For optimal evaluation, please use something that showcases the data visualization aspects of this project like jupityer notebook and or spyder in anaconda.  You also need to have pandas, numpy, scipy, matplotlib.pyplot, and various sci-kit learn modules shown in the utilities file.
# Running the tests
To evaluate the code, open the utilities.py file, and find the function you want to look at. In every function, the top half is the actual calculations being done with NumPy, and the lower half is matplotlib being used to create the charts. All of these functions are imported into the main.py file to showcase modular programming.  
# ML Model
For the ML model, I decided to stick with the trend of modular programming and write a function to handle the prep, math, and visulization.  The first third covers the preparation of getting the categorical data into dummy variables, followed my the actual training and testing using the model on the prepped data.  Finally I wanted to make a visualization of the data that allows non-technical users to see the predictions vs. reality.  This specific model/data did not have a high coefficient of determination, only being at 42%.
# Closing statement
Thank you to whoever reads this and takes the time to evaluate my code. This is my biggest project on my account here.  I put a lot of time into this analysis as it was outside of the classroom and on my own time.  This was a big step for me in modeling, as well as continuing to develop my skills in data science.  I will be looking at other forecasting methods in the future for my next small project, or maybe even an API or GUI for SQL data, and some other type of analysis with python in the background.
# Author
Nick Engelthaler
