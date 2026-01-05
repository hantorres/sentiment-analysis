## Sentiment Analysis of IMDB Movie Reviews - Python
Python Jupyter Notebook that loads in preprocessed movie review data, one dataset where reviews were stemmed, and one lemmatized. The notebook trains several models, evaluating the performance of various ML algorithms. The best model-dataset combination is chosen for further hyperparameter tuning, where the optimal model is found.
### Method
The data is first loaded using the joblib library. A cell creates a confusion matrix visualization function using matplotlib and pyplot to create interpretable visualizations for confusion matrices. The next step is to start testing models with datasets. Using Scikit-Learn, the following models are tested: Logistic Regression, Linear SVC, and Multinomial NB. The models are evaluated on accuracy because the dataset is balanced. The best performing model-dataset combination is further tuned by finding the best hyperparameters via Scikit-Learn's grid search. In this test environment, Multinomial NB was most optimal, and another library, scipy, was simply used to create a log uniform distribution for the alpha parameter. Finally, the best model is evaluated for its performance, and both the base model and tuned model parameters are displayed.

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Joblib
2) Matplotlib
3) Seaborn
4) Scikit-Learn
5) Scipy

The notebook was tested using Python version 3.13.9.

Dataset: Data was found on Kaggle https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis

### Pre-Requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

For the notebook to run properly, ensure the following files and directories exist in the same directory:
1) processed_data/
2) nlpModel.ipynb

Note: movie.csv and data_preprocessing.ipynb have been uploaded for extra information on data processing techniques used in this project. These are NOT required to train the model; however, ensure that the processed_data/ directory contains all the training files. This includes both train and test datasets for stemmed and lemmatized data (a total of 8 files). The directory also includes the TF-IDF vectorizers that created the numerical representations of movie reviews; this is also not required for the model training but included for the sake of transparency.

### Version History
V1.0 - The Jupyter Notebook is created. All cells and functions have been tested and are functional. Optimal model is found.

## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook.
### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset directory.

Using the notebook:
1) Open the notebook in the IDE.
2) Run all cells in the notebook.
3) Validate model training results using the displayed evaluation metrics, including the classification report and confusion matrix. In production, I obtained a test accuracy of 88%. Results should be similar to this. 
4) Best model can now be used in other applications or further tuned.
