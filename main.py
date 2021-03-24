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


def remove_punct(text):
    text_nopunct = ''.join([fg for fg in text if fg not in string.punctuation])
    return text_nopunct


def tokenize_text(text):
    tokens = re.split('\W+', text)
    return tokens


def remove_stopwords(tokenized_text):
    text = [w for w in tokenized_text if w not in stopwords]
    return text


df['fund_nopunct'] = df['FundStrategy'].apply(lambda x: remove_punct(x))
df['fund_tokens'] = df['FundStrategy'].apply(lambda x: tokenize_text(x.lower()))
df['fund_nostop'] = df['fund_tokens'].apply(lambda x: remove_stopwords(x))

# saving the DataFrame as a CSV file
df_csv_data = df.to_csv('C:/Users/Irfan/Desktop/Capstone/sample_data.csv', header=True)
print('\nCSV String:\n', df_csv_data, '\n', 'hello')
