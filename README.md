![patient](https://images.unsplash.com/photo-1521075486433-bf4052bb37bc?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2088&q=8)

# Sentiment Analysis
This dataset was downloaded fom this link on kaggle.com
https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets
## Context
Finding if a person is depressed from their use of words on social media can definitely help in the cure!

### Project Objective
The aim of this project is to predict depression in patients by analysing sentments from their tweets?
Sentimental Analysis can be very useful to find out depression and cure it before someone gets into serious trouble.

 ## The dataset

The dataset contains the patient records collected from a hospital. 

The "Label(depression result)" column is a target variable which has binary entries (0 or 1), indicating if the patient has depression or not.

•	Label(depression result) == 1, implies that the patient has depression.

•	Label(depression result) == 0, implies that the patient survived does not have depression

•   The message to examine column shows the message on which the Sentimental Analysis needs to be performed

•   Index column shows the ID value of a tweet

## Requirements 
Libraries used - To succesfully run this Jupyter notebook the following libraries need to be installed.

    - Python 3     - Pandas     - Scikit-Learn     - Seaborn     - Matplotlib     - Numpy - Bertopic  
    
## Data Preprocessing
Preprocessing work done on the data included:

1. Checking for missing data: there was none.
2. Tokenization of the words
3. Removal of stopwords
4. Removal of special Characters
5. Dropping of the index column
6. Lemmatizing the words
7. creating the parts os speech tag for the words

## Topic Modelling
Salient topics was identified from the sentiments to find out the top words associated with different topics

## Vectorization
TF idf vectorizer was used to vectorize the tokens

## Models 
1. Support Vector Machine

2. K Nearest Neighbors


## Results
Performance Evaluation Metric used:

1.F1 score

2.Precision

3.Recall

4.Support

5.Confusion matrix

## Imbalance Target variable

The Label was imbalanced and SMOTE technique was used to conduct oversampling and undersampling

## Model Chosen
Logistic regression was the winner with an F1_score of 0.98 for class = 0 and 0.94 for class = 0.1
