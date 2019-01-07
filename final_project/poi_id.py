# Maching learning for identifying Enron Employees who may have committed fraud 
# based on the public Enron financial and email dataset.
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi"]

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

### remove any outliers before proceeding further
data_dict.pop("TOTAL")

### collect and remove outliers from data
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))
outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
for outlier in outliers_final:
    data_dict.pop(outlier[0])
    
### create new features
### new features are: fraction_to_poi_email, fraction_from_poi_email
### divide email subset by total emails sent to/from
def dict_to_list(key, set):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][set]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][set]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi", "from_messages")

### insert new features into data_dict
count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

### store to my_dataset for easy export below
my_dataset = data_dict

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi',
'exercised_stock_options','bonus','total_stock_value'] 

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### split into labels and features
labels, features = targetFeatureSplit(data)

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### Compare SVM vs. Decision Tree Classifier vs. Naive Bayes
# Decision Tree
### Determine optimal minimum_split for Decision Tree Classifer
print ("SPLIT\tACCURACY\tPRECISION\tRECALL")
best_split = 0
max_result = 0
for i in range (2,10): 
    clf = tree.DecisionTreeClassifier(min_samples_split=i)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    if accuracy + precision + recall > max_result:
        max_result = accuracy + precision + recall
        best_split = i
    print("%d\t%0.4f\t\t%0.4f\t\t%0.4f\t" % (i,accuracy,precision,recall))
clf_tree = tree.DecisionTreeClassifier(min_samples_split=best_split)
# SVM
print "Fitting the classifier to the training set"
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
clf_svm = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf_svm = clf_svm.fit(features_train, labels_train)
print "Best estimator found by grid search:"
print clf_svm.best_estimator_
# Naive Bayes
clf_nb=GaussianNB()

clf_set = {
    "Decision_Tree": clf_tree, 
    "Naive_Bayes": clf_nb,
    "Support_Vector_Machine": clf_svm}

print ("ACCURACY\tPRECISION\tRECALL\tF1_SCORE")
for i, (clf_name, clf) in enumerate(clf_set.items()):
    print "Classifier: ", clf_name
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    f1score = f1_score(labels_test,pred)
    print("%0.4f\t\t%0.4f\t\t%0.4f\t" % (accuracy,precision,recall))
clf = clf_svm

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    # make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

    clf.fit(features_train,labels_train)
    score = clf.score(features_test,labels_test)
    print("Mean accuracy before tuning %f"% score)

    ### use manual tuning parameter min_samples_split
    clf = clf.fit(features_train,labels_train)
    pred= clf.predict(features_test)
    acc=accuracy_score(labels_test, pred)

    print ("Validating algorithm:")
    print ("Accuracy after tuning = %f"% acc)
    # Calculate precision: ratio of true positives out of all true and false positives
    print ('Precision = %lf'% precision_score(labels_test,pred))
    # Calculate recall: ratio of true positives out of true positives and false negatives
    print ('Recall = %lf'% recall_score(labels_test,pred))

### dumping classifier, dataset and features_list for verification with tester.py
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") ) 
