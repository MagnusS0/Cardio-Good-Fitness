# Predicting Customer Product Preference

<div class='tableauPlaceholder' id='viz1679330723000' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Fitness_Customer_Dashboard&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;Fitness_Customer_Dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div> 

##

In this project I created both a multiclass model and a binary model to predict the customer product preference based on their demographic and behavioral data. The data set I used for this project contains information about 180 customers who bought one of three products: TM195, TM498, and TM798. These are different models of treadmills that vary in price and features. 
The dataset can be found here: https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness

**Disclaimer:** This dataset is relatively small, which makes it challenging to draw strong conclusions about the performance of the models. Therefore, this project is mainly for the purpose of practicing and experimenting with various techniques.

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

Before building any model, I did some exploratory data analysis (EDA) to understand the distribution and relationship of the variables. I used Python and Pandas library to perform some basic statistics and Matplotlib, Seaborn and [Tableau](https://public.tableau.com/app/profile/magnus.samuelsen/viz/Fitness_Customer_Dashboard/Dashboard1) for visualization. Here are some of the findings from my EDA:

- The data set has no missing values.
- The customers who bought TM798 have higher income, fitness level, usage, miles and education than those who bought TM195 or TM498.
- The customers who bought TM195 or TM498 have similar characteristics except for miles. Those who bought TM498 expect to run more miles per week than those who bought TM195 (83 vs 88 miles)
- The customers who bought TM798 where mostly male. 

![image](https://user-images.githubusercontent.com/97634880/226214718-0b8d2823-08ca-451a-94f1-f23759b1444f.png)
![image](https://user-images.githubusercontent.com/97634880/226214988-d95f309e-4ed0-46ee-9c14-534796490ffa.png)


## Data Preprocessing

Before building any model, I also did some data preprocessing steps to prepare the data for machine learning algorithms. These steps include:

#### 1. I first encooded categorical data and created the target variable. 
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

target = cardio_df['Product'] = le.fit_transform(cardio_df['Product'])
cardio_df['Gender'] = le.fit_transform(cardio_df['Gender'])
cardio_df['MaritalStatus'] = le.fit_transform(cardio_df['MaritalStatus'])
cardio_df['Fitness'] = le.fit_transform(cardio_df['Fitness']) 
```
#### 2. I then dropped the target colum Product form the dataframe as well as Gender and MaritalStatus. (After analyzing the model, I determined that these features did not contribute to the model).
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

#### 3. Next was scaling the data to avoid bias and make sure that models like e.g. KNN could be run. 
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

#### 4. Beacuse the data set was unbalanced I determind to use under smapling to reduse the majority classes (TM194 and TM498).
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(scaled_df, target)
```
#### 5. Then only one step was left and that was splitting the data into a train and test set (70/30).
```python
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
```

## Multiclass Model

The first model I built was a multiclass model that predicts which product (TM195, TM498, or TM798) a customer would buy based on their features.

### Model building
I first created a pipline that can be used to benchmark differnt classifiers against each other. 
The following classifiers was used:
- KNN
- Decision Tree
- Linear Regression
- SVM
```python
#Create a list with all the models
model_pipeline = []
model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(DecisionTreeClassifier(random_state=42))
model_pipeline.append(LogisticRegression(random_state=42))
model_pipeline.append(SVC(random_state=42))


#Create the list of models and the accuracy of each model
model_list = ['KNN', 'Decision Tree', 'Logistic Regression', 'SVM']
traning_accuracy_list = []
accuracy_list = []
mean_corss_val=[]

for model in model_pipeline:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    traning_accuracy_list.append(model.score(X_train, y_train))
    accuracy_list.append(metrics.accuracy_score(y_test, y_pred))
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_corss_val.append(scores.mean())

#And now compare the performance of the models in a dataframe
result_df = pd.DataFrame({'Model': model_list,'Traning Accuracy':traning_accuracy_list ,'Accuracy': accuracy_list, 'Cross Validation': mean_corss_val})
```

Becasue of the small size of the datset more complex classifiers like Random Forest and Gradient Boosting were not used.
To evaluate the trained models I used Accuracy and Mean Cross Validation. 

| Model | Training Accuracy | Accuracy | Cross Validation |
|-------|-------------------|----------|------------------|
| KNN | 0.7976619 | 0.583333 | 0.644118 |
| Decision Tree | 0.892857 | 0.611111 | 0.750735 |
| Linear Regression | 0.785714 | 0.694444 | 0.701471 |
| SVM | 0.738095 | 0.611111 | 0.714706 |

### Hyperparameter tuning
To improve the performance of the multiclass model, I performed hyperparameter tuning using GridSearchCV from the Scikit-learn library.
```python
#Grid search for logistic regression
param_grid = {
    'C': [0, 0.01, 0.1, 0,5, 1, 5, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'random_state': [42],
    'multi_class': ['multinomial'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'max_iter': [100, 200, 300],
    }

# Create a logistic regression classifier
logreg = LogisticRegression()

# Create a grid search object
grid_search = GridSearchCV(logreg, param_grid, cv=5,)
# Fit the grid search object to the data
grid_search.fit(X_train, y_train)
```

### Results
After training and tuning, the logistic regression model perfomece the best, the following performance was achieved:
### Logistic Regression Model

- Training Accuracy: 0.7857
- Accuracy: 0.75
- Cross Validation Score: 0.7382

|     | Precision | Recall | F1-score | Support |
|-----|-----------|--------|----------|---------|
|  0  |   0.70    |  0.58  |   0.64   |    12   |
|  1  |   0.62    |  0.67  |   0.64   |    12   |
|  2  |   0.92    |  1.00  |   0.96   |    12   |
|-----|-----------|--------|----------|---------|
| Avg |   0.75    |  0.75  |   0.75   |    36   |

The classification report shows that the model has a high precision and recall for class 2 (TM798), which could be due to the fact that this class has more distinct features, making it easier to distinguish from the others. The precision and recall for classes 0 (TM195) and 1 (TM498) are lower, indicating that the model may have more difficulty distinguishing between these two classes. In the EDA part it is very clear that these two classes have many overlapping features. Overall the model achieved an accuracy of 0.75 and a cross validation score of 0.7382 on the test set, indicating that it generalizes well to unseen data.

## Binary Model
Because of the many similarities between TM 195 and TM478 I decided to merge them and build a binary classification model to predict whether a customer will buy the high-end treadmill (TM798) or not.
```python
# Merge product TM195 and TM 498 into one category
cardio_df['Product'] = cardio_df['Product'].replace(['TM195', 'TM498'], 'TM195_TM498')
cardio_df['Product'].unique()
```
I also had to change `sampling_strategy` to make sure that the model did not loose information and still capture the true distribution of the data after merging the two classes. I tested this by looking at weighted avg F-1 score that increased e.g. from 0.87 to 0.97 for the SVM model. 
```python
rus = RandomUnderSampler(random_state=42, sampling_strategy= {0: 80, 1: 40})
X_res, y_res = rus.fit_resample(scaled_df, target)
X_res.shape
```
### Model building
Again I created a pipline that can be used to benchmark differnt classifiers against each other. 
The following classifiers was used:
- KNN
- Decision Tree
- Linear Regression
- SVM

| Model             | Training Accuracy | Accuracy | Cross Validation |
|-------------------|-------------------|----------|-----------------|
| KNN               | 0.988095         | 0.944444 | 0.976471       |
| Decision Tree     | 1.0               | 0.972222   | 0.928676        |
| SVM               | 1.0          | 0.972222 | 0.988235       |
| Logistic Regression | 1.0      | 0.972222 | 0.988235       |

As expected, the binary classification models achieves higher accuracy than the multi-class one.
This make sense as the model no longer has to deal with distinguishing between TM195 and TM498.

### Hyperparameter tuning
To improve the performance of the already well performing binary models, I performed hyperparameter tuning using GridSearchCV. 
### Results
After training and tuning the Super Vector Machine (SVM) model perfomece the best, the following performance was achieved:

- Training Accuracy: 1.0
- Accuracy: 0.9722
- Cross Validation Score: 1.0

|    | precision   | recall   | f1-score   | support   |
|---|-----------|--------|----------|---------|
|  0 |     0.96    |    1.00  |    0.98    |     24    |
|  1 |     1.00    |    0.92  |    0.96    |     12    |
|----------|---------|-------|-----------|-------------|
|accuracy |            |          |    0.97    |     36    |
|macro avg |    0.98    |    0.96  |    0.97    |     36    |
|weighted avg | 0.97    |    0.97  |    0.97    |     36    |


Overall, this model achieved high accuracy of 0.97 and cross-validation score of 1.0 on the test set, meaning that the model makes predictions with confidence and consistency. The results indicate that the SVM model can accurately classify customers based on their treadmill preferences. However, if we look at the F-1 score for TM798, it actually did not improve from the multi-class model, indicating that the multi-class model also did a great job at identifying TM798 customers. This also tells me that the accuracy improvements overall mostly come from merging the two product categories, eliminating the difficulty with distinguishing TM195 and TM498. However, it should be noted that the dataset used for this model is relatively small, so these results may not generalize well to larger or more diverse datasets.


## Evaluation and Applicablility
- The models can be used by marketing or sales teams to target customers based on their preferences and offer them personalized recommendations or promotions.
- If a company can collect data on potential customers, the models can predict which products they are most likely to purchase. ML models are capable of making these predictions much faster than humans, making them a valuable tool for marketing and sales teams looking to optimize their targeting strategies.
- The multi-class model does a good job at identifying TM978 customers but struggled with differentiating between TM195 and TM498.
- The binary model has better overall accuracy, but it is not better at identifying TM978. Therefore, it is up to preference whether a marketer or sales person should use the multi-class or binary model.
- However, both models may not be robust enough to handle new or unseen data, as the dataset used for training and testing was small and limited. Therefore, the models should be validated on larger and more diverse datasets before deploying them in real-world scenarios.

## Conclusion
In this project, I built two models, a multiclass model and a binary model, to predict the customer product preference based on their demographic and behavioral data.

I started with some basic EDA and built a dashboard with Tableau. 
Then did some neccesary preprocessing, before splitting the data into a traning and test set.
I then applied different models to the classification problem and used grid search to perform hyperparameter tuning.
The final product is a multiclass logistic regression model and a SVM binary classification model.

Overall, this project was mainly for the purpose of practicing and experimenting with various techniques, given the small dataset size. Nonetheless, the results from the models can be used as a starting point for further exploration and analysis. This project represents a valuable exercise in data science and machine learning for me, highlighting the importance of careful data preparation, model selection, and evaluation.
