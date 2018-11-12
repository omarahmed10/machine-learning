import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math

"""
    Load Dataset
"""

print(os.listdir("./"))
mainPath = '../input/san-francisco-crime-classification/'
#Load Data with pandas, and parse the first column into datetime
train=pd.read_csv('all/train.csv', parse_dates = ['Dates'])
test=pd.read_csv('all/test.csv', parse_dates = ['Dates'])

"""
    DataSet treatment
"""

def transform(data):
    alpha = (2*math.pi)/24
    (hours,minutes,seconds) = (data.Dates.dt.hour, data.Dates.dt.minute, data.Dates.dt.second)
    time = hours + (minutes / 60) + (seconds / 3600)
    TimeX = math.cos(alpha) * time
    TimeY = math.sin(alpha) * time
    TimeX = pd.DataFrame(data={'TimeX':TimeX})
    TimeY = pd.DataFrame(data={'TimeY':TimeY})
    del data['Dates']
    # convert Day of week to Cartesian coordinates
    from sklearn import preprocessing
    daysMapper = preprocessing.LabelEncoder()
    dayNumber = daysMapper.fit_transform(data.DayOfWeek)
    beta = (2*math.pi)/7
    dayX = [ math.cos(beta*d) for d in dayNumber ]
    dayY = [ math.sin(beta*d) for d in dayNumber ]
    dayX = pd.DataFrame(data={'dayX':dayX})
    dayY = pd.DataFrame(data={'dayY':dayY})
    del data['DayOfWeek']
    # delete PdDistrict and Address
    del data['PdDistrict']
    del data['Address']
    transformed_data = pd.concat([data,dayX,dayY,TimeX,TimeY],axis=1)
    return transformed_data

transformed_data = transform(train)
transformed_test = transform(test)
transformed_test = transformed_test.drop(['Id'],axis=1)


#from sklearn.cross_validation import train_test_split
# split training and validation datasets
#training, validation = train_test_split(transformed_data, train_size=0.8)
# split again validation into two to generate the validation and testing datasets
# validation, testing = train_test_split(validation, train_size=0.5)


"""
    DataSet Clustering
    
"""
def cluster(data):
    from sklearn.cluster import KMeans
    from collections import Counter
    result = pd.DataFrame()
    data_size = data.shape[0]
    data_final_size = 8000
    min_data_size = 10
    all_cat = list(set(data.Category))
    indx = 0
    for cat in all_cat:
        print('Category '+cat+'...')
        elements = data.loc[data.Category == cat]
        number_centroids = max(min_data_size, round(elements.shape[0] * data_final_size / data_size))
        if number_centroids > elements.shape[0]:
            number_centroids = elements.shape[0]
        number_centroids = int(number_centroids)
        elements = elements.drop(['Category','Descript','Resolution'],axis=1)
        print('Applying kmeans with '+str(number_centroids)+' centroids...')
        kmeans = KMeans(n_clusters=number_centroids, random_state=0).fit(elements)
        # Find the cluster centers
        centers = kmeans.cluster_centers_
        predict = kmeans.predict(elements)
        indices = list(range(indx,indx+number_centroids))
        n_samples = list(Counter(predict).values())
        centroids = []
        for i in range(0,len(centers)):
            centroids.append(np.append(centers[i], [cat, n_samples[i]]))
        centroids = pd.DataFrame(centroids,columns=['X', 'Y', 'dayX', 'dayY', 'TimeX', 'TimeY', 'Category', 'N_samples'],index= indices)
        print('new centroid data',centroids.shape,centroids.head())
        result = result.append(centroids)
        print('new results shape',result.shape)
        print(result.head())
        indx += number_centroids
    return result

#clusterd_traing_data = cluster(transformed_data)
#clusterd_traing_data.to_csv('reduced_train.csv', sep=',', index=False)



"""
    Evaluating Models....
"""

from time import time
def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    results = {}
    
    start = time() # Get start time
    learner.fit(X_train, y_train)
    end = time() # Get end time
    results['train_time'] = end - start
        
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    
    results['pred_time'] = end - start
    
    start = time() # Get start time
    predictions_poba_test = learner.predict_proba(X_test)
    predictions_poba_train = learner.predict_proba(X_train)
    end = time() # Get end time
    
    results['pred_proba_time'] = end - start
    
    from sklearn.metrics import fbeta_score , accuracy_score , log_loss
    results['acc_train'] = accuracy_score(y_train, predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = fbeta_score(y_train, predictions_train , beta=0.5, average='micro')
    results['f_test'] = fbeta_score(y_test, predictions_test , beta=0.5, average='micro')
    
    print(y_test.shape,predictions_poba_test.shape)
    results['logloss_train'] = log_loss(y_train, predictions_poba_train)
    results['logloss_test'] = log_loss(y_test, predictions_poba_test)
    
    # Success
    print("{} finish training".format(learner.__class__.__name__))
      
    # Return the results
    return results

def evaluate(learners, X_train, y_train, X_test, y_test):
    # Collect results on the learners
    results = {}
    for clf in learners:
       clf_name = clf.__class__.__name__
       results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)
    
    # print results
    for model in results:
        model_res = results[model]
        print ("model: {}".format(model))
        print ("train time:\t{}\nacc_train:\t{}\nf_train:\t{}".format(model_res['train_time'],model_res['acc_train'],model_res['f_train']))
        print ("======================================================================================")
        print ("pred time:\t{}\nacc_test:\t{}\nf_test:\t{}".format(model_res['pred_time'],model_res['acc_test'],model_res['f_test']))
        print ("======================================================================================")
        print ("pred proba time:\t{}\nlogloss_train:\t{}\nlogloss_test:\t{}".format(model_res['pred_proba_time'],model_res['logloss_train'],model_res['logloss_test']))
    return results

"""
    load and split the reduced data
"""
reduced_data = pd.read_csv('all/reduced_train.csv')
features = reduced_data.drop(['Category','N_samples'],axis=1)
from sklearn import preprocessing
catMapper = preprocessing.LabelEncoder()
categories_number = catMapper.fit_transform(reduced_data.Category)
categories = catMapper.inverse_transform(list(range(0,categories_number.max()+1)))
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, categories_number, test_size=0.3, random_state=50)


"""
    training with KNN
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=600).fit(features, categories_number)
print('start Predicting....')
answer = pd.DataFrame(columns=categories)
indx = 0
for i in range(100000, 900000, 100000): 
    indices = list(range(indx,indx+100000))
    probs = knn.predict_proba(transformed_test[indx:i])
    answer = answer.append(pd.DataFrame(probs,columns=categories,index= indices))
    print('end predicting from',indx ,'to' ,i , answer.shape)
    indx += 100000
indices = list(range(indx,84262))
probs = knn.predict_proba(transformed_test[indx:])
answer = answer.append(pd.DataFrame(probs,columns=categories,index= indices))
print('end predicting from',indx ,'to' ,884262, answer.shape)
answer.to_csv('../working/answer_knn.csv', sep=',', index_label = 'Id')


"""
    Finding the best model
"""

from sklearn.neural_network import MLPClassifier
learner_knn = KNeighborsClassifier(n_neighbors=600,n_jobs=-1, weights='distance')
learner_MLP = MLPClassifier(learning_rate='invscaling', shuffle=True)
evaluate([learner_MLP],X_train, y_train ,X_test, y_test)


probs = learner_MLP.predict_proba(transformed_test)
answer = pd.DataFrame(probs,columns=categories)
answer.to_csv('all/answer_MLP.csv', sep=',', index_label = 'Id')
           





