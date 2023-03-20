# Predicting Customer Product Preference

<div class='tableauPlaceholder' id='viz1679330723000' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Fitness_Customer_Dashboard&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div> 

##

In this project I created both a multiclass model and a binary model to predict the customer product preference based on their demographic and behavioral data. The data set I used for this project contains information about 180 customers who bought one of three products: TM195, TM498, and TM798. These are different models of treadmills that vary in price and features. 
The dataset can be found here: https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness

## Data Description

The data set has 8 features and 1 target variable. The features are:

- Miles: The average number of miles the customer expects to run each week.
- Fitness: The self-rated fitness level of the customer on a scale of 1 to 5.
- Usage: The number of times the customer expects to use the treadmill each week.
- Income: The annual household income of the customer in dollars.
- Age: The age of the customer in years.
- Education: The number of years of education completed by the customer.
- Gender: Male or Female
- Marital Status: Single or Partnered

The target variable is **Product**, which has three possible values: TM195, TM498, and TM798.

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

The first model I built was a multiclass model that predicts which product (TM195, TM498, or TM798) a customer would buy based on their features.
### Preprocessing
1. To build this model, I first encooded categorical data and dropped the target value from the dataframe. 
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

target = cardio_df['Product'] = le.fit_transform(cardio_df['Product'])
cardio_df['Gender'] = le.fit_transform(cardio_df['Gender'])
cardio_df['MaritalStatus'] = le.fit_transform(cardio_df['MaritalStatus'])
cardio_df['Fitness'] = le.fit_transform(cardio_df['Fitness']) 
```
2. I then dropped Gender and MaritalStatus. (After analyzing the model, I determined that these features did not contribute to the model).
```python
cardio_df.drop(['Product'], axis=1, inplace=True)
#Drop categories that are not needed
cardio_df.drop(['Gender', 'MaritalStatus'], axis=1, inplace=True)
cardio_df.head()
```
| Age | Education | Usage | Fitness | Income | Miles |
| --- | --- | --- | --- | --- | --- |
| 18 | 14 | 3 | 3 | 29562 | 112 |
| 19 | 15 | 2 | 2 | 31836 | 75 |
| 19 | 14 | 4 | 2 | 30699 | 66 |
| 19 | 12 | 3 | 2 | 32973 | 85 |
| 20 | 13 | 4 | 1 | 35247 | 47 |

3. Next was scaling the data to avoid bias and make sure that models like KNN could be run. 
```python
# Scale the data using StandardScaler
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(cardio_df)
scaled_df = pd.DataFrame(scaled_df, columns=cardio_df.columns)
scaled_df.head()
```
| Age | Education | Usage | Fitness | Income | Miles |
| --- | --- | --- | --- | --- | --- |
| -1.558146 | -0.974987 | -0.421117 | 0.720443 | -1.467585 | 0.170257 |
| -1.413725 | -0.354854 | -1.345520 | -0.325362 | -1.329438 | -0.545143 |
| -1.413725 | -0.974987 | 0.503286 | -0.325362 | -1.398512 | -0.719159 |
| -1.413725 | -2.215254 | -0.421117 | -0.325362 | -1.260365 | -0.351792 |
| -1.269303 | -1.595120 | 0.503286 | -1.371166 | -1.122218 | -1.086527 |

4. Beacuse the data set was unbalanced I determind to use under smapling to recuse the majority classes (TM194 and TM498).
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(scaled_df, target)
```
5. Then only one step was left and that was splitting the data into a train and test set (70/30).
```python
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
```
### Model building
I first created a pipline that can be used to benchmark differnt classifiers against each other. 
The following classifiers was used:
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

To evaluate the trained models I used Accuracy and Mean Cross Validation. 

| Model | Accuracy | Cross Validation |
|-------|----------|-----------------|
| KNN | 0.583333 | 0.644118 |
| Decision Tree | 0.555556 | 0.786029 |
| Random Forest | 0.611111 | 0.701471 |
| Gradient Boosting | 0.666667 | 0.786029 |

Based on these accuracys I decided to go on with Decision Tree and Gradient Boosting for hyperparameter tuning. 


