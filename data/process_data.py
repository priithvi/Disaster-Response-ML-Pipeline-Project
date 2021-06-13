import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This will load the datasets for the exercise - messages.csv and categories.csv
    We will also merge the two datasets together into a single pandas dataframe
    
    Parameters:
    messages_filepath: the location of the messages.csv file
    categories_filepath: the location of the categories.csv file
    
    Returns:
    df: a dataframe which is created by merging messages.csv and categories.csv
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = "inner", on = "id")
    
    return df

    pass


def clean_data(df):
    
    """
    The merged data is cleaned
    
    Parameters:
    df: unclean dataframe
    
    Returns:
    df: cleaned dataframe
    
    """
    
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories[:1]

    
    
    cols = []
    for i in range(len(row.columns)):
        cols.append(str(row[i][0]))

    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    cols1 = []
    for i in cols:
        cols1.append(i[0:len(i[0])-3])

    
    category_colnames = cols1

    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand = True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    df = df[df['related'] != 2]
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


    pass


def save_data(df, database_filename):
    """
    This will save the df dataframe into sqlite database
    
    Parameters:
    df: pandas dataframe
    
    database_filename: database name where the file is saved
    
    Returns:
    
    file in database
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df_db1', engine, index = False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()