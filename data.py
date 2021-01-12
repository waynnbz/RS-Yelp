import json
import pandas as pd
import numpy as np

bizPATH = 'data/yelp_dataset/yelp_academic_dataset_business.json'
revPATH = 'data/yelp_dataset/yelp_academic_dataset_review.json'
userPATH = 'data/yelp_dataset/yelp_academic_dataset_user.json'
covidPATH = 'data/covid_19_dataset_2020_06_10/yelp_academic_dataset_covid_features.json'

topLabels = ['restaurants', 'nightlife', 'bars', 'food',
       'american (traditional)', 'american (new)', 'sushi bars',
       'cocktail bars', 'japanese', 'sports bars', 'pubs', 'barbeque',
       'breakfast & brunch', 'juice bars & smoothies', 'wine bars',
       'burgers', 'seafood', 'sandwiches', 'lounges', 'asian fusion',
       'steakhouses', 'coffee & tea', 'wine & spirits', 'beer',
       'event planning & services']

topStates = ['AZ', 'NV', 'ON', 'OH', 'NC', 'PA', 'QC', 'AB', 'WI']



class Business:
    """
    Business class object for easy accessing information.
    """

    def __init__(self, business_id, name, address, coordinate, stars, review_count, attributes, categories):
        self.business_id = business_id
        self.name = name
        self.address = address
        self.coordinate = coordinate
        self.stars = stars
        self.review_count = review_count
        self.attributes = attributes
        self.categories = categories

    # override default format to print
    def __repr__(self):
        return (
            f"Address: {self.address}\nCoordinate: {self.coordinate}\nStars: {self.stars}\nCategories: {self.categories}"
        )


def BusinessDecoder(obj):
    """
    Decoder from json to Business object
    """
    address = f"{obj['address']} {obj['city']} {obj['state']} {obj['postal_code']}"
    coordinate = (obj['latitude'], obj['longitude'])

    return Business(obj['business_id'], obj['name'], address, coordinate, obj['stars'], obj['review_count'],
                    obj['attributes'], obj['categories'])


def bizCatFilter(path, cat='bar'):
    """
    extracting a dict of businesses of the input category,
    dict implemented in hashtable with mostly O(1) look up time

    :param path: Business data file path
    :param cat: Interested business category
    :return:
        bizDict: a diction {key: business id, value: business objects}
        A mapper from business name to business id

    """

    bizDict = {}
    name2idMapper = {}

    with open(path, 'r') as infile:
        for line in infile:

            data = json.loads(line)

            if data['categories']:
                if cat.lower() in data['categories'].lower():
                    # select only businesses with more than ten reviews for interpretability
                    if data['review_count'] >= 10:
                        name2idMapper[data['name']] = data['business_id']
                        bizDict[data['business_id']] = BusinessDecoder(data)

    return bizDict, name2idMapper


def reviewMatrix(path, bizDict, start_year, threshold):
    """
    generating a review matrix of n_biz x n_user with values of rating
    :param bizDict: dictionary with target business id as keys
    :param path: review data file path
    :param start_year: from which the sampling starts
    :param threshold: minimum number of review counts
    :return: sampled review matrix dataframe
    """

    reviews = list()
    with open(path, 'r') as infile:
        for i, line in enumerate(infile):

            data = json.loads(line)

            if data['business_id'] in bizDict:
                user_id = data['user_id']
                business_id = data['business_id']
                stars = data['stars']
                date = data['date']

                reviews.append([user_id, business_id, stars, date])

    df = pd.DataFrame(reviews, columns=['user_id', 'business_id', 'stars', 'date'])
    df.date = pd.to_datetime(df.date)
    df = df[df['date'] > start_year]
    df = df.drop(columns='date')
    df = df.drop_duplicates(subset=['business_id', 'user_id'], keep='last')

    threshold = 10
    df = df.groupby('user_id').filter(lambda x: len(x) >= threshold)
    df = df.groupby('business_id').filter(lambda x: len(x) >= threshold)

    revs = df.pivot(
        index='business_id',
        columns='user_id',
        values='stars'
    ).fillna(0)

    return revs


def businessData(revs, path, topLabels, topCities):
    # set is hashed thus faster
    bizIds = set(revs.index)

    bizData = []
    with open(path, 'r') as infile:
        for i, line in enumerate(infile):

            data = json.loads(line)

            if data['business_id'] in revs.index:
                business_id = data['business_id']
                review_count = data['review_count']

                if data['state'] in topStates:
                    state = data['state']
                else:
                    state = 'Other_State'


                if data['categories']:
                    flag = False
                    for label in data['categories'].lower().split(', '):
                        if label in topLabels:
                            flag = True
                    if flag:
                        categories = data['categories']
                    else:
                        categories = 'Other_Category'

                bizData.append([business_id, review_count, state, categories])

    df = pd.DataFrame(bizData, columns=['business_id', 'review_count', 'state', 'categories'])
    df.set_index('business_id', drop=True, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['state'], prefix='S')
    df[topLabels] = 0
    df['Other_Category'] = 0

    def label(row, topLabels):
        OC = True
        for l in row['categories'].lower().split(', '):
            if l in topLabels:
                row[l] = 1
                OC = False
        if OC:
            row['Other_Category'] = 1

        return row

    df = df.apply(lambda row: label(row, topLabels), axis=1).drop('categories', axis=1)
    df.review_count = np.log(df.review_count)

    return df


def addCovidFeature(bizData, path):
    bizIds = set(bizData.index)

    with open(path) as infile:
        for i, line in enumerate(infile):

            data = json.loads(line)

            if data['business_id'] in bizIds:
                if data['highlights'] != 'FALSE':
                    bizData.loc[data['business_id'], 'hasHighlights'] = 1
                else:
                    bizData.loc[data['business_id'], 'hasHighlights'] = 0

                if data['delivery or takeout'] != 'FALSE':
                    bizData.loc[data['business_id'], 'delivery or takeout'] = 1
                else:
                    bizData.loc[data['business_id'], 'delivery or takeout'] = 0

    return bizData


def userData(path, revs):
    userIds = set(revs.columns)

    uData = []
    with open(path, 'r') as infile:
        for i, line in enumerate(infile):

            data = json.loads(line)

            if data['user_id'] in userIds:

                user_id = data['user_id']
                review_count = data['review_count']
                yelping_since = data['yelping_since']
                friends = len(data['friends'].split())
                feedback = data['useful'] + data['funny'] + data['cool']
                fans = data['fans']
                if len(data['elite']) == 0:
                    elite = 0
                else:
                    elite = len(data['elite'].split(','))
                total_compliments = data['compliment_hot'] + data['compliment_more'] + data['compliment_profile'] + \
                                    data['compliment_cute'] + data['compliment_list'] + data['compliment_note'] + data[
                                        'compliment_plain'] + data['compliment_cool'] + data['compliment_funny'] + data[
                                        'compliment_writer'] + data['compliment_photos']

                uData.append([user_id, review_count, yelping_since, friends, feedback, fans, elite, total_compliments])

    df = pd.DataFrame(uData,
                      columns=['user_id', 'review_count', 'yelping_since', 'friends', 'feedback', 'fans', 'elite',
                               'total_compliments'])
    df = df.set_index('user_id', drop=True)
    df.review_count = np.log(df.review_count)
    df.yelping_since = np.log(pd.to_datetime(df.yelping_since).apply(lambda x: (pd.to_datetime('2020-01-01') - x).days))
    df.friends = np.log(df.friends)
    df.feedback = np.log1p(df.feedback)
    df.fans = np.log1p(df.fans)
    df.total_compliments = np.log1p(df.total_compliments)

    return df

