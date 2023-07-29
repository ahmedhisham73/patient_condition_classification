#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Reading data**

# In[2]:


import pandas as pd

# define file paths
test_data_path = "/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.csv"
train_data_path = "/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv"

# read the .csv files
test_data = pd.read_csv(test_data_path)
train_data = pd.read_csv(train_data_path)


# In[5]:


print(train_data.head())
print(train_data.info())


# In[6]:


# print the column names
print(train_data.columns)


# In[7]:


# Check for null values in train_data
null_values = train_data.isnull().sum()

# Print the count of null values per column
print(null_values)


# In[8]:


# Get unique values and their counts in the 'condition' column
condition_counts = train_data['condition'].value_counts()

# Print the unique values and their counts
print(condition_counts)


# In[9]:


# Get unique values and their counts in the 'condition' column
condition_counts = train_data['condition'].value_counts()

# Filter conditions with counts more than 5000
frequent_conditions = condition_counts[condition_counts > 5000]

# Print the filtered conditions and their counts
print(frequent_conditions)


# In[10]:


# Define the conditions to keep
conditions_to_keep = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne']

# Filter the DataFrame to only include the specified conditions
filtered_data = train_data[train_data['condition'].isin(conditions_to_keep)]

# Now filtered_data only includes rows where the 'condition' is in conditions_to_keep


# In[11]:


# Define the conditions and features to keep
conditions_to_keep = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne']
features_to_keep = ['uniqueID', 'drugName', 'condition', 'review', 'rating', 'date']

# Filter the DataFrame to only include the specified conditions and features
filtered_data = train_data[train_data['condition'].isin(conditions_to_keep)][features_to_keep]


# In[12]:


# Get the count of non-null values in each feature
feature_counts = filtered_data.count()

# Print the counts
print(feature_counts)


# In[13]:


# Get the shape of the DataFrame
data_shape = filtered_data.shape

# Print the shape
print(data_shape)


# In[15]:


# Select 'review', 'condition', and 'rating' columns and create a new DataFrame
data_first_stage = filtered_data[['review', 'condition', 'rating']]


# In[16]:


# Get the shape of the DataFrame
data_shape = data_first_stage.shape

# Print the shape
print(data_shape)


# In[23]:





# In[20]:





# In[24]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')

import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Define a function to handle all text preprocessing
def preprocess_text(text):
    # Apply the pipeline to your text
    doc = nlp(text)
    
    # Tokenize, lower case, and lemmatize the text
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply the preprocessing function to the 'review' column
data_first_stage['review'] = data_first_stage['review'].apply(preprocess_text)



# In[25]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Concatenate all reviews into one string
all_reviews = ' '.join(data_first_stage['review'])

# Create and generate a word cloud image
wordcloud = WordCloud().generate(all_reviews)

# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram
plt.figure(figsize=(10,5))
sns.histplot(data_first_stage['rating'], bins=10, kde=True)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Create a boxplot
plt.figure(figsize=(5,10))
sns.boxplot(y=data_first_stage['rating'])
plt.title('Rating Boxplot')
plt.ylabel('Rating')
plt.show()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
condition_counts = data_first_stage['condition'].value_counts()
sns.barplot(x=condition_counts.index, y=condition_counts.values, alpha=0.8)
plt.title('Condition Frequency')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Condition', fontsize=12)
plt.xticks(rotation=90)
plt.show()



# In[29]:


import random

# Get a random index
random_index = random.choice(data_first_stage.index)

# Print the review at the random index
print(data_first_stage.loc[random_index, 'review'])


# In[30]:


print(data_first_stage['rating'].describe())


# In[ ]:





# **now lets scale the rating**

# In[33]:


from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()

# create a new DataFrame from data_first_stage
data_second_stage = data_first_stage.copy()

# apply the transformation on the new DataFrame
data_second_stage.loc[:, 'rating'] = scaler.fit_transform(data_second_stage[['rating']])




# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.histplot(data_second_stage['rating'], bins=10, kde=True)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[37]:





# In[40]:


# Check for missing values
missing_values = data_first_stage['condition'].isna().sum()
print(f"Number of missing values in 'condition': {missing_values}")

# If there are missing values, drop the corresponding rows
if missing_values > 0:
    data_first_stage = data_first_stage.dropna(subset=['condition'])

# Check again for missing values
missing_values = data_first_stage['condition'].isna().sum()
print(f"Number of missing values in 'condition' after dropping: {missing_values}")


# In[54]:


from tensorflow.keras import backend as K
K.clear_session()


# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Convert your categories into numerical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data_first_stage['condition'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_first_stage['review'], labels, test_size=0.2, random_state=42)

# Create a pipeline that first creates bag of word representation then applies the classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier()),
])

# Train the classifier
model.fit(X_train, y_train)

# Test the classifier
predictions = model.predict(X_test)

# Print a classification report
print(classification_report(y_test, predictions))



# In[58]:


from joblib import dump

# Save the model to a file
dump(model, 'model_rndomFor.joblib') 


# In[65]:


# Load the test data
test_data_raw = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.csv')

# Filter the conditions
conditions_to_include = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne']
# Filter the conditions and make a copy of the dataframe
test_data = test_data_raw[test_data_raw['condition'].isin(conditions_to_include)].copy()

# Now you can apply the changes to test_data without affecting test_data_raw
test_data['review'] = test_data['review'].apply(preprocess_text)
test_data['condition'] = label_encoder.transform(test_data['condition'])


# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the counts of each condition
condition_counts = test_data['condition'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=condition_counts.index, y=condition_counts.values, alpha=0.8)
plt.title('Counts of Each Condition in the Test Data')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Condition', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[67]:


# Use the model to predict sentiments for the test data
test_predictions = model.predict(test_data['review'])

# Convert the probabilities to class labels
predicted_labels = [np.argmax(prediction) for prediction in test_predictions]

# Now predicted_labels contains the predicted labels for the test data


# In[68]:


# Convert the numeric labels back to original classes
predicted_conditions = label_encoder.inverse_transform(predicted_labels)
actual_conditions = label_encoder.inverse_transform(test_data['condition'])

# Create a dataframe that includes the predicted and actual labels
results = pd.DataFrame({'Predicted Condition': predicted_conditions, 'Actual Condition': actual_conditions})

# Print the dataframe
print(results.head())


# In[69]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
accuracy = accuracy_score(actual_conditions, predicted_conditions)
precision = precision_score(actual_conditions, predicted_conditions, average='weighted')
recall = recall_score(actual_conditions, predicted_conditions, average='weighted')
f1 = f1_score(actual_conditions, predicted_conditions, average='weighted')

print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')


# In[70]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a pipeline that first creates bag of word representation then applies the classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# Train the classifier
model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(X_test)

# Print a classification report
print(classification_report(y_test, y_pred))

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')


# In[71]:


import joblib

# Save the model to a file
joblib.dump(model, 'naive_bayes_model.pkl')

# Later you can load the model from the file with:
# model = joblib.load('naive_bayes_model.pkl')


# In[73]:


# Select a random review and its corresponding condition from the test data
random_index = np.random.choice(test_data.index)
random_review = test_data.loc[random_index, 'review']
actual_condition = label_encoder.inverse_transform([test_data.loc[random_index, 'condition']])

print("Original review: \n", random_review)

# Use the model to predict the sentiment of this review
prediction = model.predict([random_review])

# The output of the model is the predicted class label
predicted_condition = label_encoder.inverse_transform([prediction])

print("\nPredicted condition: ", predicted_condition)
print("Actual condition: ", actual_condition)


# In[74]:


from sklearn.linear_model import LogisticRegression

# Create a pipeline that first creates bag of word representation then applies the classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear')),
])

# Train the classifier
model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(X_test)

# Print a classification report
print(classification_report(y_test, y_pred))

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')


# In[76]:


# Select a random review and its corresponding condition from the test data
random_index = np.random.choice(test_data.index)
random_review = test_data.loc[random_index, 'review']
actual_condition = label_encoder.inverse_transform([test_data.loc[random_index, 'condition']])

print("Original review: \n", random_review)

# Use the model to predict the sentiment of this review
prediction = model.predict([random_review])

# The output of the model is the predicted class label
predicted_condition = label_encoder.inverse_transform([prediction])

print("\nPredicted condition: ", predicted_condition)
print("Actual condition: ", actual_condition)


# In[77]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Transform the training data
X_train_transformed = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_transformed = vectorizer.transform(X_test)

# Define the three models to use in the ensemble
model1 = make_pipeline(TfidfVectorizer(), LogisticRegression())
model2 = make_pipeline(TfidfVectorizer(), MultinomialNB())
model3 = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())

# Create the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', model1), ('nb', model2), ('dt', model3)], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = ensemble_model.predict(X_test)

# Print a classification report
print(classification_report(y_test, y_pred))

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')


# In[78]:


import joblib

# Save the model to a file
joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Later you can load the model from the file with:
# ensemble_model = joblib.load('ensemble_model.pkl')


# In[79]:


# Load the model from the file
ensemble_model = joblib.load('/kaggle/working/ensemble_model.pkl')


# In[80]:


# Use the model to predict the conditions of the reviews in the test set
predicted_conditions = ensemble_model.predict(test_data['review'])

# Convert the numeric labels back to the original conditions
predicted_conditions = label_encoder.inverse_transform(predicted_conditions)

# Create a DataFrame to compare the actual and predicted conditions
comparison_df = pd.DataFrame({
    'Actual Condition': label_encoder.inverse_transform(test_data['condition']),
    'Predicted Condition': predicted_conditions
})

print(comparison_df)


# In[81]:





# In[ ]:




