from data import *
from sklearn.neighbors import NearestNeighbors


bizPATH = 'data/yelp_dataset/yelp_academic_dataset_business.json'
revPATH = 'data/yelp_dataset/yelp_academic_dataset_review.json'
userPATH = 'data/yelp_dataset/yelp_academic_dataset_user.json'
covidPATH = 'data/covid_19_dataset_2020_06_10/yelp_academic_dataset_covid_features.json'


def knn_model(mat_revs):
    """
    kth nearest neighbor model
    :param mat_revs: Review matrix, n_biz x n_user
    :return: trained model
    """
    # making model class
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    # fit the sparse matrix
    knn.fit(mat_revs)

    return knn




def train():






if __name__ == '__main__':
    train()