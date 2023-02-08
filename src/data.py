# import gc
import re
# import string
# import operator
# from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.graph_objs as go
# import plotly.tools as tls
import plotly.offline as py
# import tokenization
from nltk import word_tokenize 
from nltk.util import ngrams
# py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


def data_visualization(df):
    # pass the data as a dataframe
    # data visualization
    print(df.head())

    # cheack if any null values
    print(df.isnull().sum())

    #check for the duplicate
    print(df.duplicated().sum())

    # check for how many total no of duplicates and total number of not duplicate
    print(df['is_duplicate'].value_counts())
    print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
    df['is_duplicate'].value_counts().plot(kind='bar')
    plt.savefig('total_dup_nondup.png')

    #check how many duplicate questions are there

    qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
    print('Number of unique questions',np.unique(qid).shape[0])
    x = qid.value_counts()>1
    print('Number of questions getting repeated',x[x].shape[0])

    # Repeated questions histogram

    plt.hist(qid.value_counts().values,bins=160)
    plt.yscale('log')
    plt.savefig('each_que_no_rep.png')

def data_engineering(data):
    # adding the length (each indivuadial letters) of each questions to the data frame
    data['q1_len'] = data['question1'].str.len() 
    data['q2_len'] = data['question2'].str.len()

    # adding the total number of words of each questions in the data frame
    data['q1_num_words'] = data['question1'].apply(lambda row: len(str(row).split(" ")))
    data['q2_num_words'] = data['question2'].apply(lambda row: len(str(row).split(" ")))

    # common_words
    data['overlap_count'] = (data.apply(lambda r: set(r['question1'].split(" ")) &
                                             set(r['question2'].split(" ")),                               
                                            axis=1)).str.len()

    # count the total number of words from both question 1 and 2
    # method 1
    # def total_words(row):
    # w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    # w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    # return (len(w1) + len(w2))
    # method 2
    data['total_num_words'] = data['q1_num_words']+data['q2_num_words']

    data['word_share'] = round(data['overlap_count']/data['total_num_words'],2)

    # finding common_word_count min and max, common_stop_count min and max, common_token_count min and max, last_word_eq, first_word_eq
    def fetch_token_features(row):
        
        q1 = row['question1']
        q2 = row['question2']

        SAFE_DIV = 0.0001 

        STOP_WORDS = stopwords.words("english")
        
        token_features = [0.0]*8
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()
        
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        
        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        
        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))
        
        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))
        
        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        
        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        
        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        
        return token_features

    token_features = data.apply(fetch_token_features, axis=1)

    data["cwc_min"]       = list(map(lambda x: x[0], token_features))
    data["cwc_max"]       = list(map(lambda x: x[1], token_features))
    data["csc_min"]       = list(map(lambda x: x[2], token_features))
    data["csc_max"]       = list(map(lambda x: x[3], token_features))
    data["ctc_min"]       = list(map(lambda x: x[4], token_features))
    data["ctc_max"]       = list(map(lambda x: x[5], token_features))
    data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    data["first_word_eq"] = list(map(lambda x: x[7], token_features))

    # finding the absoulute length diff, mean len, and longest subtr ratio
    def fetch_length_features(row):
        q1 = row['question1']
        q2 = row['question2']
        
        length_features = [0.0]*3
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()
        
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return length_features
        
        # Absolute length features
        length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
        
        #Average Token Length of both Questions
        length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
        
        strs = list(distance.lcsubstrings(q1, q2))
        # length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
        
        return length_features

    length_features = data.apply(fetch_length_features, axis=1)

    data['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    data['mean_len'] = list(map(lambda x: x[1], length_features))
    data['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

    # fuzzy features

    def fetch_fuzzy_features(row):
    
        q1 = row['question1']
        q2 = row['question2']
        
        fuzzy_features = [0.0]*4
        
        # fuzz_ratio
        fuzzy_features[0] = fuzz.QRatio(q1, q2)

        # fuzz_partial_ratio
        fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

        # token_sort_ratio
        fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

        # token_set_ratio
        fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

        return fuzzy_features

    fuzzy_features = data.apply(fetch_fuzzy_features, axis=1)

    # Creating new feature columns for fuzzy features
    data['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    data['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    data['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    data['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

    return data

def data_analysis(dataset):

    # Analysis the features of q1_len
    sns.displot(dataset['q1_len'])
    print('minimum characters',dataset['q1_len'].min())
    print('maximum characters',dataset['q1_len'].max())
    print('average num of characters',int(dataset['q1_len'].mean()))
    plt.savefig('q1_len_data_analysis.png')

    # Analysis the features of q2_len
    sns.displot(dataset['q2_len'])
    print('minimum characters',dataset['q2_len'].min())
    print('maximum characters',dataset['q2_len'].max())
    print('average num of characters',int(dataset['q2_len'].mean()))
    plt.savefig('q2_len_data_analysis.png')

    # Analysis the features of q1_num_words
    sns.displot(dataset['q1_num_words'])
    print('minimum words',dataset['q1_num_words'].min())
    print('maximum words',dataset['q1_num_words'].max())
    print('average num of words',int(dataset['q1_num_words'].mean()))
    plt.savefig('q1_num_words_data_analysis.png')

    # Analysis the features of q2_num_words
    sns.displot(dataset['q2_num_words'])
    print('minimum words',dataset['q2_num_words'].min())
    print('maximum words',dataset['q2_num_words'].max())
    print('average num of words',int(dataset['q2_num_words'].mean()))
    plt.savefig('q2_num_words_data_analysis.png')

    # common words
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['overlap_count'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['overlap_count'],label='duplicate')
    plt.savefig('common_words_data_analysis.png')

    # total words
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['total_num_words'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['total_num_words'],label='duplicate')
    plt.savefig('total_words_data_analysis.png')

    # word share
    sns.distplot(dataset[dataset['is_duplicate'] == 0]['word_share'],label='non duplicate')
    sns.distplot(dataset[dataset['is_duplicate'] == 1]['word_share'],label='duplicate')
    plt.savefig('word_share_data_analysis.png')
    # print(dataset.head())

    # comparision image between common token, word, stop word count min
    sns.pairplot(dataset[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_ctc_cwc_csc_min.png')

    # comparision image between common token, word, stop word count max
    sns.pairplot(dataset[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_ctc_cwc_csc_max.png')

    # comparision image between last word common equal, first word equal
    sns.pairplot(dataset[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_lastw_firstw.png')

    # comparision image between mean length, abs len, longest substring ratio
    sns.pairplot(dataset[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_meanlen_abslen_lgstsbstrat.png')

    # comparision image between fuzzy ratio, fuzzy partial ratio, token sort ratio and token set ratio
    sns.pairplot(dataset[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')
    plt.savefig('comparision_between_fuzzy.png')

def ngram_features(row):
    
    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    ngram_feature = [0.0]*2
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    bigram_que1 = set([ngrams(q1_tokens, 2)]) 
    bigram_que2 = set([ngrams(q2_tokens, 2)]) 

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    
    # Get the common non-stopwords from Question pair
    count_of_bigram1 = len(bigram_que1)
    count_of_bigram2 = len(bigram_que2)

    
    ngram_feature[0] = count_of_bigram1
    ngram_feature[1] = count_of_bigram2
    
    return ngram_feature

def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q 

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    print(data.shape)
    data = data.dropna()

    data['question1'] = data['question1'].apply(preprocess)
    data['question2'] = data['question2'].apply(preprocess)

    print('Data Set Shape = {}'.format(data.shape))
    print('Data Set Memory Usage = {:.2f} MB'.format(data.memory_usage().sum() / 1024**2))


    data_visualization(data)
    dataset = data_engineering(data)
    dataset.to_csv('dataset.csv')
    print(dataset.shape)
    data_analysis(dataset)

    ngram_features = data.apply(ngram_features, axis=1)
    print(np.shape(ngram_features))
    print(ngram_features)

