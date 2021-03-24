import numpy as np
import pandas as pd
import re
import nltk
import string

data = pd.read_excel('C:/Users/Irfan/Desktop/Capstone/proj_data.xlsx')

df = data[['Name', 'new_strategy_extracted', 'Target Variable']]
df.columns = ['Name', 'FundStrategy', 'Target Variable']
df_sample = df[0:5]

stopwords = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


df['fund_cleaned'] = df['FundStrategy'].apply(lambda x: clean_text(x.lower()))
df['fund_lemmatized'] = df['fund_cleaned'].apply(lambda x: lemmatizing(x))


# saving the DataFrame as a CSV file
df_csv_data = df.to_csv('C:/Users/Irfan/Desktop/Capstone/sample_data.csv', header=True)
print('\nCSV String:\n', df_csv_data, '\n', 'hello')
