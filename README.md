# Predicting Customer Product Preference

<div class='tableauPlaceholder' id='viz1679263027500' style='position: relative'><noscript><a href='https:&#47;&#47;www.kaggle.com&#47;datasets&#47;saurav9786&#47;cardiogoodfitness'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Fitness_Customer_Dashboard&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div> 


In this project I created both a multiclass model and a binary model to predict the customer product preference based on their demographic and behavioral data. The data set I used for this project contains information about 180 customers who bought one of three products: TM195, TM498, and TM798. These are different models of treadmills that vary in price and features. 
The dataset can be found here: https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness

## Data Description

The data set has 8 features and 1 target variable. The features are:

- Product: The product code of the treadmill purchased by the customer.
- Miles: The average number of miles the customer expects to run each week.
- Fitness: The self-rated fitness level of the customer on a scale of 1 to 5.
- Usage: The number of times the customer expects to use the treadmill each week.
- Income: The annual household income of the customer in dollars.
- Age: The age of the customer in years.
- Education: The number of years of education completed by the customer.
- Gender: Male or Female
- Marital Status: Single or Partnered

The target variable is Product, which has three possible values: TM195, TM498, and TM798.

## Data Exploration

Before building any model, I did some exploratory data analysis (EDA) to understand the distribution and relationship of the variables. I used Python and Pandas library to perform some basic statistics and Matplotlib, Seaborn and Tableau for visualization. Here are some of the findings from my EDA:

- The data set has no missing values.
- The customers who bought TM798 have higher income, fitness level, usage, and education than those who bought TM195 or TM498.
- The customers who bought TM195 or TM498 have similar characteristics except for miles and age. Those who bought TM498 expect to run more miles per week than those who bought TM195 (83 vs 88 miles)
- The customers who baught TM798 where mostly male. 

![image](https://user-images.githubusercontent.com/97634880/226214718-0b8d2823-08ca-451a-94f1-f23759b1444f.png)
![image](https://user-images.githubusercontent.com/97634880/226214988-d95f309e-4ed0-46ee-9c14-534796490ffa.png)


## Data Preprocessing

Before building any model, I also did some data preprocessing steps to prepare the data for machine learning algorithms. These steps include:

- Covert object data types to categorical data type using `.astype('category')`
- Encoding categorical variables (Product, Gender, Marital Status, Fitness Level (1-5)) into numerical values using LabelEncoder from scikit-learn library, and creating `target = cardio_df['Product'] = le.fit_transform(cardio_df['Product'])`
- Dropping the Product column from the table 
- Splitting the data into training and testing sets using train_test_split from scikit-learn library
- Scaling numerical variables (Miles, Fitness, Usage, Income, Age, Education) using StandardScaler from scikit-learn library

## Multiclass Model

The first model I built was a multiclass model that predicts which product (TM195, TM498, or TM798) a customer would buy based on their features. To build this model, 

     

