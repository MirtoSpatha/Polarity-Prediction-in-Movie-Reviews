# Positive and Negative Sentiment Prediction in Movie Reviews

In this project, we used a dataset of positive and negative movie reviews to predict the sentiment linked with each review (positive or negative). 
The dataset creation was based on a 5-star rating that the reviewers gave. 

We used **Supervised Machine Learning** Methods because they obtain promising results in Classification problems and help the new engineers understand the basic principles of Machine Learning. We also created our project in a notebook style to improve readability and we developed it in the Google Collab Environment to avoid memory issues and to help both collaborators update the project in the same notebook.

The methods used to predict the sentiment were **Bayesian Classification, Logistic Regression, Support Vector Machine and an LSTM Deep Neural Network**. All the methods had an accuracy above 75%.

Below, you can see how we structured our project and what were the main parts of our work:
## 1. Preprocessing
First, we cleaned the data by removing html, urls, stop words, non alphabet characters and upper case letters.
Then, we corrected the spelling of the words and normalized them using Stemming and Lemmatization. 
Finally, we splitted the dataset into training and test sets and vectorized the words with CountVectorizer and TfidfVectorizer.
## 2. Data Understanding
We need to understand what the dataset contains in order to help the models make more accurate predictions. For this reason, we created graphs that showed the most common words in the dataset as well as in positive and negative reviews, the most frequent bigrams and the sentimental words.
## 3. Naive Bayes Model
We trained a Naive Bayes Model with and without Laplacian Smoothing to see whether the negative probabilities affect the predictions. We showed that Tdidf Vectorizer when used with smoothing gives accurate results in 75% of the cases. 
## 4. Logistic Regression Model
We trained a Logistic Regression Model with and without L2 penalty to manage overfitting. We showed that Tdidf Vectorizer when used with l2 penalty gives accurate results in 86% of the cases.
## 5. Support Vector Machine Model
We trained a Support Vector Machine Model with poly and linear kernel to see if our data are linearly separable or not. We showed that Tdidf Vectorizer when used with linear kernel gives accurate results in 88% of the cases.
## 6. LSTM Model
We trained an LSTM Model with 100 batch_size, 40 epochs and validation_split = 0.3. We showed that it gives accurate results in 79% of the cases.
## 7. Results Comparison
All methods give mostly accurate results with the best method being the SVM Model.
## 8. Future Work
* Data Analysis of falsely classified reviews (in terms of spelling, unrecognised words by the models
  and other data patterns).
* Testing with more review or general datasets (combine results).
* Omitting non sentimental words in the pre-processing step.


This project was written in **Python** and in collaboration with **Thibaut Le Marre** ( https://github.com/thibaut-lm ).
