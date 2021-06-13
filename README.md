## Disaster Reponse Machine Learning Pipeline Project

## Description
This project is part of Udacity's Data Science Nano Degree online program. In the project I analyze text data provided by Figure 8 to classify them into type of disaster response they correspond to. For doing this, natural language processing, and machine learning model are used.

The project is based on real world data of short messages sent by users during time of natural disasters such as flood, hurricance, etc. These messages need to be directed to the right disaster response department for appropriate quick remedial action. 

## Instructions on how to run
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Description of files
1. process.py: An ETL pipeline what does:
    - loads messages.csv and categories.csv datasets
    - merges them together and does data cleaning
    - saves the combined and cleaned file to sql database
   
2. train_classifier.py: An machine learning pipeline that doesL
    - loads sql database file saved from step 1 above into pandas dataframe
    - tokenizes text data in the file using nltk library
    - creates a machine learning pipeline to train a model using GridSearchCV
    - saves the final best model into a pickle file

3. run.py: Runs a Flask Web app that visualizes the results from the above machine learning model prediction


## Licensing, Authors, and Acknowledgements
Data was provided by Udacity.com and orginally is from Figure Eight


