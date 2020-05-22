import pandas as pd


# Loads the sample data and produces a csv file that can
# be used by the build/train pipeline script.
def create_sample_data_csv():
    df = pd.read_csv('D:/VSCodeProjects/AzureRecommender/'
                     'data/aml_recommender.csv')
    df.to_csv('aml_recommender.csv', index=False)
