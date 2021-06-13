## Project Name: Disaster Reponse Machine Learning Pipeline Project

## Description
This project is part of Udacity's Data Science Nano Degree online program. In the project I analyze text data provided by Figure 8 to classify them into type of disaster response they correspond to. For doing this, natural language processing, and machine learning model are used.

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
1. process.py 


## Data source and types of data
The data is downloaded from Udacity's Workspace. The following datasets are provided:
1. customers.csv: This file contains list and some characteritics of the company's customers
2. azdias.csv: This file contains the same characteritics as of the customers.csv file for Germany's general population which are not the company's customers yet
3. DIAS Information Levels - Attributes 2017.csv: This file contains the description of each of the variables/customer characteristics present in the customers.csv and azdias.csv files
4. DIAS Attributes - Values 2017.csv: This file contains the description of each of the variables present in the customers.csv and azdias.csv files, along with the dewscription of each of the categorical values taken by the variables
5. mailout_train.csv: It contains a list of customer ids of the company's customers along with a flag "RESPONSE" to indicate whether they responded to a marketing campaign in the past or not. Customer attributes are also present
6. mailout_test.csv: It contains a list of new customer ids of the company's customers. It also contains customer attributes, but the "RESPONSE" indicator is not present


## Code file
The Python code file created used to do all the analysis is available by the name: Arvato_analysis.ipynb

## Terms and Conditions
In addition to Udacity's Terms of Use and other policies, your downloading and use of the AZ Direct GmbH data solely for use in the Unsupervised Learning and Bertelsmann Capstone projects are governed by the following additional terms and conditions. The big takeaways:

You agree to AZ Direct GmbH's General Terms provided below and that you only have the right to download and use the AZ Direct GmbH data solely to complete the data mining task which is part of the Unsupervised Learning and Bertelsmann Capstone projects for the Udacity Data Science Nanodegree program.

You are prohibited from using the AZ Direct GmbH data in any other context.

You are also required and hereby represent and warrant that you will delete any and all data you downloaded within 2 weeks after your completion of the Unsupervised Learning and Bertelsmann Capstone projects and the program.

If you do not agree to these additional terms, you will not be allowed to access the data for this project.
The full terms are provided in the workspace below. You will then be asked in the next workspace to agree to these terms before gaining access to the project, which you may also choose to download if you would like to read in full the terms.

These same exact terms are provided in the next workspace, where you will be asked to accept the terms prior to gaining access to the data.

## Licensing, Authors, and Acknowledgements
Technical repose of the project is also create on Medium here: https://mail-prithvi.medium.com/arvato-financial-services-segmentation-and-predictive-modeling-6552eb64e20d
This same report is present in the repository by the name: Arvato_analysis.pdf
Apart from those licensing information, anyone if free to use and reproduce the code as they like.



