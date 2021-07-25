#import the libraries
import pandas as pd , numpy as np
from datetime import datetime,date
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import time
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.util import ngrams
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

user_final_rating= None
initialized=None
productReviews=None
SentimentModel=None
wordVectorizer=None

def loadAllModelsAndData():
    global productReviews
    if productReviews is None:
        productReviews = pd.read_csv('dataset/sample30.csv' , encoding='latin-1')
    global user_final_rating
    if user_final_rating is None:
        start = time.time()
        user_final_rating = pickle.load(open("pickle/user_final_rating.pkl", 'rb'))
        print(f"Model loaded from file user_final_rating.pkl in time: {time.time() - start} secs")
    global SentimentModel
    if SentimentModel is None:
        file_name = "pickle/BestSentimentAnalysisModel_XGBoost_Tuned.pkl"
        start = time.time()
        SentimentModel = pickle.load(open(file_name, 'rb'))
        print(f"Model loaded from file {file_name} in time: {time.time() - start} secs")
    global wordVectorizer
    if wordVectorizer is None:
        file_name = "pickle/wordVectorizer.pkl"
        start = time.time()
        wordVectorizer = pickle.load(open(file_name, 'rb'))
        print(f"wordVectorizer loaded from file {file_name} in time: {time.time() - start} secs")
    
def init():
    global initialized
    print("Initializing- Loading all required models")
    loadAllModelsAndData()
    initialized = True
    print("Initialized - All reqired Models")
    
    

#Assign stop words
def setStopWords():
  stopwords_custom = set(stopwords.words('english'))
  return stopwords_custom

### Define a function to clean the text from noizy text data such as: 

    #trimming spacing
    #removing redudant punctuation
    #substituting text to a plain form e.g.: won't -> will not
    #remove stopwords except negative words
    #Lemmatize the Phrases

def text_clean(df,stopwords_custom):

    reviewTextTitleSeries = df.reviews_text_title.copy()
    #print(reviewTextTitleSeries)
    reviewTextTitleSeries= reviewTextTitleSeries.apply(lambda x: x.lower())
    clean = re.compile('<.*?>') # Remove HTML tag
    reviewTextTitleSeries= reviewTextTitleSeries.apply(lambda x: re.sub(clean, '', x))

    reviewTextTitleSeries = reviewTextTitleSeries.apply(lambda x: re.sub(r"http\S+", "", x))  # removing URls
    reviewTextTitleSeries = reviewTextTitleSeries.apply(lambda x: re.sub(r"www\S+", "", x))


    reviewTextTitleSeries = reviewTextTitleSeries.str.replace('\\*', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace('\\/', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace('\\\\', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace('-', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'/', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'``', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'`', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"''", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r",", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"\.$", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r":", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"# ", '#', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r";", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"?", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"=", ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("...", ' ', regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("..", ' ', regex=False)

    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'LRB', ' ', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'RRB', ' ', regex=True)
    
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]on't", 'will not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]eren't", 'were not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]asn't", 'was not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]ouldn't", 'would not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]oesn't", 'does not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[I|i]sn't", 'is not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[C|c]ouldn't", 'could not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]idn't", 'did not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]asn't", 'has not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]aven't", 'have not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]on't", 'do not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[A|a]in't", "not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[N|n]eedn't", "need not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[A|a]ren't", "are not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[S|s]houldn't", "should not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]adn't", "had not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[C|c]an't", "can not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[M|m]ightn't", "might not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[M|m]ustn't", "must not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[S|s]han't", "shall not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[N|n]t", "not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]o n't", 'will not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]ere n't", 'were not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]as n't", 'was not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[W|w]ould n't", 'would not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]oes n't", 'does not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[I|i]s n't", 'is not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[C|c]ould n't", 'could not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]id n't", 'did not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]as n't", 'has not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]ave n't", 'have not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[D|d]o n't", 'do not', regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[A|a]i n't", "not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[N|n]eed n't", "need not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[A|a]re n't", "are not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[S|s]hould n't", "should not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"[H|h]ad n't", "had not", regex=True)
    
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"ain", "are not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"doesn", "does not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"hasn", "has not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"didn", "did not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"mightn", "might not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"shouldn", "should not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"hadn", "had not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"wasn", "was not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"wouldn", "would not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"mustn", "must not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"needn", "need not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"weren", "were not", regex=True)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r"shan", "shall not", regex=True)

    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(" 's", " ", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'s", "", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'ve", "have", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'d", "would", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'ll", "will", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'m", "am", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'n", "and", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'re", "are", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace("'til", "until", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(" ' ", " ", regex=False)
    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(" '", " ", regex=False)

    reviewTextTitleSeries = reviewTextTitleSeries.str.replace(r'[ ]{2,}', ' ', regex=True)
    
    # Remove Stopwords and also lemmatize the phrases
    lem=WordNetLemmatizer()
    reviewTextTitleSeries=reviewTextTitleSeries.apply(word_tokenize)
    reviewTextTitleSeries=reviewTextTitleSeries.apply(lambda x: ' '.join(([lem.lemmatize(w, pos='v') for w in x if w not in stopwords_custom])))
    return reviewTextTitleSeries


def get_recommendations(username):
    global user_final_rating
    d = user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    d=d.reset_index()
    return(d['id'])



def filterProductBySentiment(recommendedProducts):
    # Disabling Stopwords and text clean for review text title before passing to Model while selecting top 5 products,to avoid heroku 30 sec time out
    #stopwords_custom=setStopWords()
    #stopwords_custom = [ word for word in stopwords_custom if 'n\'t' not in word and 'not' not in word and 'no' not in word ]
    productSentimentPercentageDict={}
    global productReviews
    global wordVectorizer
    global SentimentModel
    for product in recommendedProducts:
        dfProductReview=productReviews[(productReviews['id']==product)][['reviews_text','reviews_title','name']]
        dfProductReview['reviews_text']=dfProductReview['reviews_text'].replace(np.NaN," ")
        dfProductReview['reviews_title']=dfProductReview['reviews_title'].replace(np.NaN," ")
        dfProductReview['reviews_text_title']=dfProductReview['reviews_text']+" "+dfProductReview['reviews_title']
        dfProductReview.drop(['reviews_text','reviews_title'],axis=1,inplace=True)
        # Disabling Stopwords and text clean for review text title before passing to Model while selecting top 5 products,to avoid heroku 30 sec time out
        #dfProductReview['reviews_text_title']=text_clean(dfProductReview,stopwords_custom)
        tfidfVector=wordVectorizer.transform(dfProductReview['reviews_text_title'])
        train_features = pd.DataFrame(tfidfVector.toarray(),columns=wordVectorizer.get_feature_names())
        train_features.reset_index(drop=True,inplace=True)
        dfX = train_features
        dfProductReview['sentimentPredict']=SentimentModel.predict(dfX)
        productPositiveSentimentPercentage=(dfProductReview[(dfProductReview['sentimentPredict']==1)]['sentimentPredict'].size/dfProductReview['sentimentPredict'].size)*100
        productSentimentPercentageDict[dfProductReview['name'].iloc[0]]=productPositiveSentimentPercentage
    filteredProductList=list({k: v for k, v in sorted(productSentimentPercentageDict.items(), key=lambda item: item[1],reverse=True)}.items())[:5]
    filteredProductList=dict(filteredProductList)
    #print(filteredProductList)
    filteredProductList={k: v for k, v in sorted(filteredProductList.items(), key=lambda item: item[1],reverse=True)}
    return(filteredProductList)

def getFinalRecommendaions(username):
    global initialized
    if initialized is None:
        print(f"initialized is {initialized}")
        init()
    print("Recommending 5 best product for "+str(username))
    recommendedProducts=list(get_recommendations(username))
    print(recommendedProducts)
    filteredProductList=filterProductBySentiment(recommendedProducts)
    print(filteredProductList)
    return filteredProductList


def main(username):
    filteredProductList=getFinalRecommendaions(username)
    print(filteredProductList)

if __name__ == "__main__":
    username = 'tammy'
    main(username)
