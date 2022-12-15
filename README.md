The goal is to predict the duration of medical sick leaves based on data of previous leaves.

Each row of data is one previous leave of one worker, and includes:

    - Age of the worker 
    - Time that the worker has been in the company
    - Sex of the worker
    - Indicator of whether the sick leave is a relapse
    - Number of sick leaves that the worker has had up to the moment of being sick 
    - Number of individual leaves that the worker has been until the moment of being on leave 
    - CNAE (activity category) of the company in which that worker works 
    - Worker group of tax contribution 
    - Type of contract of the worker
    - Type of disease that the doctor has diagnosed according to ICD9 categorization
    - Size of the company in which the worker works 
    - Total number of accumulated sick leaves that the company has 
    - Day of the week on which the leave occurred 
    - Week of the year in which the leave occurred 
    - Month of the year in which the leave occurred 
    - Postal code of the person who is on leave
    - Weather the person is working on several jobs

    - Sick leave duration  <- what to predict

The interface has been designed with streamlit, and its files are in the folder smarthealing_app/.
It can be run through the following url : https://smart-healing.streamlit.app/

Both classification and regression models (XGBoost) have been locally trained and saved with the file smarthealing_mod/model/model.py.
The raw data has been preprocessed with the file smarthealing_mod/preprocessor.py.

The API has been designed using FastAPI. Its files are in the folder smarthealing_api/.
The API retrieves the saved trained models, and preprocess the new input again with preprocessor.py.
The API works in Google Cloud Run with a Docker image created using the file dockerfile.
