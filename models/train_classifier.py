import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import BaseEstimator, CountVectorizer, TfidfTransformer, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
nltk.download('wordnet')


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.multioutput import MultiOutputClassifier
import pickle

from sklearn.metrics import confusion_matrix, classification_report




def load_data(database_filepath):
    """
    This will load the data from sql database
    
    Parameters:
    database_filepath: The location of the data file
    
    Returns:
    The dataset split into X, Y, containing the "training features" and the "dependent variables", respectively, 
    and category names
    """
    
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from df_db1", con = engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    
    return X, Y, category_names
    
    pass


def tokenize(text):
    """
    This will process text data and convert it into tokenized words
    
    Parameters:
    Text string 
    
    Returns:
    Tokenized words after cleaning
    """
    
    a = word_tokenize(text.lower().strip())
    b = [i for i in a if i not in stopwords.words("english")]
    
    c = []
    for i in b:
        c.append(WordNetLemmatizer().lemmatize(i))
        
    return c

    pass


def build_model():
    """
    It is a multioutput classification machine learning model pipeline to 
    classify text data into any of 36 categories
    
    Parameters:
    None
    
    Returns:
    cv: A model which can predict text data into 36 categories on any new data
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {'vect__max_df': [0.5, 0.75], 
    #              'vect__ngram_range': [1, 5],
    #              'tfidf__sublinear_tf': [True, False],
    #              'tfidf__smooth_idf': [True, False],
    #               'tfidf__norm': ['l1', 'l2', None]

                 'model__estimator__max_depth': [2, 3],
                 'model__estimator__n_estimators': [5, 10],
                 }

    cv = GridSearchCV(pipeline, parameters)
    
    return cv
    
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model based on Precision, Recall, and f1-score
    
    Parameters:
    model: a machine learning model
    X_test: text messages
    Y_test: category label for each of the text messages in X_test
    category_names = names of the categories in Y_test
    """
   

    pred_tuned = model.predict(X_test)
    print(classification_report(Y_test.values, pred_tuned, target_names = category_names))
    
    pass


def save_model(model, model_filepath):
    """
    Saves the machine learning model 
    
    Parameters:
    model: the machine learning model to be saved
    model_filepath: local where the file will be saved
    
    Returns:
    None
    """
    
#     filename = 'finalized_model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()