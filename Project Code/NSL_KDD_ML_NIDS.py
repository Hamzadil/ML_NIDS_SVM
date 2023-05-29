import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
             "land","wrong_fragment","urgent","hot","num_failed_logins",
             "logged_in","num_compromised","root_shell","su_attempted","num_root",
             "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
             "is_host_login","is_guest_login","count","srv_count","serror_rate",
             "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
             "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

train = pd.read_csv('NSL_KDD_Train.csv',header=None, names = col_names)
test = pd.read_csv('NSL_KDD_Test.csv',header=None, names = col_names)

print("-----------------------------------------------------------------------------------------------------------------------------------")

print('Dimensions of the Training set:',train.shape)
print('Dimensions of the Test set:',test.shape)

print(train.head(5))

print("-----------------------------------------------------------------------------------------------------------------------------------")

print('Label distribution Training set:')
print(train['label'].value_counts())
print()
print('Label distribution Test set:')
print(test['label'].value_counts())

print("-----------------------------------------------------------------------------------------------------------------------------------")

print()
print('Training set:')
for col_name in train.columns:
    if train[col_name].dtypes == 'object' :
        unique_cat = len(train[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print()
print('Distribution of categories in service:')
print(train['service'].value_counts().sort_values(ascending=False).head())

print()
print('Test set:')
for col_name in test.columns:
    if test[col_name].dtypes == 'object' :
        unique_cat = len(test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print("-----------------------------------------------------------------------------------------------------------------------------------")

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

train_categorical_values = train[categorical_columns]
test_categorical_values = test[categorical_columns]

print()
print(train_categorical_values.head())

print("-----------------------------------------------------------------------------------------------------------------------------------")

unique_protocol=sorted(train.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
print(unique_protocol2)

unique_service=sorted(train.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
print(unique_service2)

unique_flag=sorted(train.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
print(unique_flag2)

dumcols=unique_protocol2 + unique_service2 + unique_flag2

unique_service_test=sorted(test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

print("-----------------------------------------------------------------------------------------------------------------------------------")

train_categorical_values_enc=train_categorical_values.apply(LabelEncoder().fit_transform) # type: ignore

print(train_categorical_values.head())
print('--------------------')
print(train_categorical_values_enc.head())

print()

# test set
test_categorical_values_enc=test_categorical_values.apply(LabelEncoder().fit_transform) # type: ignore

print(test_categorical_values.head())
print('--------------------')
print(test_categorical_values_enc.head())

print("-----------------------------------------------------------------------------------------------------------------------------------")

enc = OneHotEncoder(categories='auto')
train_categorical_values_encenc = enc.fit_transform(train_categorical_values_enc)
train_cat_data = pd.DataFrame(train_categorical_values_encenc.toarray(),columns=dumcols) # type: ignore

# test set
test_categorical_values_encenc = enc.fit_transform(test_categorical_values_enc)
test_cat_data = pd.DataFrame(test_categorical_values_encenc.toarray(),columns=testdumcols) # type: ignore

print(train_cat_data.head())

print("-----------------------------------------------------------------------------------------------------------------------------------")

trainservice=train['service'].tolist()
testservice= test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
print(difference)

print("-----------------------------------------------------------------------------------------------------------------------------------")

for col in difference:
    test_cat_data[col] = 0

print(train_cat_data.shape)    
print(test_cat_data.shape)
     
print("-----------------------------------------------------------------------------------------------------------------------------------")

newtrain=train.join(train_cat_data)
newtrain.drop('flag', axis=1, inplace=True)
newtrain.drop('protocol_type', axis=1, inplace=True)
newtrain.drop('service', axis=1, inplace=True)

# test data
newtest=test.join(test_cat_data)
newtest.drop('flag', axis=1, inplace=True)
newtest.drop('protocol_type', axis=1, inplace=True)
newtest.drop('service', axis=1, inplace=True)

print(newtrain.shape)
print(newtest.shape)

print("-----------------------------------------------------------------------------------------------------------------------------------")

labeltrain=newtrain['label']
labeltest=newtest['label']


# change the label column
newlabeltrain=labeltrain.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeltest=labeltest.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})


# put the new label column back
newtrain['label'] = newlabeltrain
newtest['label'] = newlabeltest

print("-----------------------------------------------------------------------------------------------------------------------------------")

to_drop_DoS = [0,1]
to_drop_Probe = [0,2]
to_drop_R2L = [0,3]
to_drop_U2R = [0,4]


DoS_train=newtrain[newtrain['label'].isin(to_drop_DoS)];
Probe_train=newtrain[newtrain['label'].isin(to_drop_Probe)];
R2L_train=newtrain[newtrain['label'].isin(to_drop_R2L)];
U2R_train=newtrain[newtrain['label'].isin(to_drop_U2R)];


DoS_test=newtest[newtest['label'].isin(to_drop_DoS)];
Probe_test=newtest[newtest['label'].isin(to_drop_Probe)];
R2L_test=newtest[newtest['label'].isin(to_drop_R2L)];
U2R_test=newtest[newtest['label'].isin(to_drop_U2R)];


print('Train:')
print('Dimensions of DoS:' ,DoS_train.shape)
print('Dimensions of Probe:' ,Probe_train.shape)
print('Dimensions of R2L:' ,R2L_train.shape)
print('Dimensions of U2R:' ,U2R_train.shape)
print()
print('Test:')
print('Dimensions of DoS:' ,DoS_test.shape)
print('Dimensions of Probe:' ,Probe_test.shape)
print('Dimensions of R2L:' ,R2L_test.shape)
print('Dimensions of U2R:' ,U2R_test.shape)

print("-----------------------------------------------------------------------------------------------------------------------------------")

X_DoS = DoS_train.drop('label',axis=1)
Y_DoS = DoS_train.label

X_Probe = Probe_train.drop('label',axis=1) 
Y_Probe = Probe_train.label

X_R2L = R2L_train.drop('label',axis=1) 
Y_R2L = R2L_train.label

X_U2R = U2R_train.drop('label',axis=1) 
Y_U2R = U2R_train.label

# test set
X_DoS_test = DoS_test.drop('label',axis=1) 
Y_DoS_test = DoS_test.label

X_Probe_test = Probe_test.drop('label',axis=1) 
Y_Probe_test = Probe_test.label

X_R2L_test = R2L_test.drop('label',axis=1) 
Y_R2L_test = R2L_test.label

X_U2R_test = U2R_test.drop('label',axis=1) 
Y_U2R_test = U2R_test.label



colNames=list(X_DoS)
colNames_test=list(X_DoS_test)
     


from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 

scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe)

scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L)

scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R) 

# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 

scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 

scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test) 

scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test)



# SVM Classifier for DoS category
svm_dos = SVC(kernel='linear')
rfe_dos = RFE(estimator=svm_dos, n_features_to_select=13, step=1)
rfe_dos.fit(X_DoS, Y_DoS.astype(int))
X_rfe_dos = rfe_dos.transform(X_DoS)
selected_features_dos = [colNames[i] for i in range(len(rfe_dos.support_)) if rfe_dos.support_[i]]

# SVM Classifier for Probe category
svm_probe = SVC(kernel='linear')
rfe_probe = RFE(estimator=svm_probe, n_features_to_select=13, step=1)
rfe_probe.fit(X_Probe, Y_Probe.astype(int))
X_rfe_probe = rfe_probe.transform(X_Probe)
selected_features_probe = [colNames[i] for i in range(len(rfe_probe.support_)) if rfe_probe.support_[i]]

# SVM Classifier for R2L category
svm_r2l = SVC(kernel='linear')
rfe_r2l = RFE(estimator=svm_r2l, n_features_to_select=13, step=1)
rfe_r2l.fit(X_R2L, Y_R2L.astype(int))
X_rfe_r2l = rfe_r2l.transform(X_R2L)
selected_features_r2l = [colNames[i] for i in range(len(rfe_r2l.support_)) if rfe_r2l.support_[i]]

# SVM Classifier for U2R category
svm_u2r = SVC(kernel='linear')
rfe_u2r = RFE(estimator=svm_u2r, n_features_to_select=13, step=1)
rfe_u2r.fit(X_U2R, Y_U2R.astype(int))
X_rfe_u2r = rfe_u2r.transform(X_U2R)
selected_features_u2r = [colNames[i] for i in range(len(rfe_u2r.support_)) if rfe_u2r.support_[i]]

# Print selected features for each category
print("Features selected for DoS:", selected_features_dos)
print()
print("Features selected for Probe:", selected_features_probe)
print()
print("Features selected for R2L:", selected_features_r2l)
print()
print("Features selected for U2R:", selected_features_u2r)
print()

# Print shape of transformed feature matrices
print("Shape of X_rfeDoS:", X_rfe_dos.shape)
print("Shape of X_rfeProbe:", X_rfe_probe.shape)
print("Shape of X_rfeR2L:", X_rfe_r2l.shape)
print("Shape of X_rfeU2R:", X_rfe_u2r.shape)


Y_DoS_pred=svm_dos.predict(X_DoS_test)
Y_Probe_pred=svm_probe.predict(X_Probe_test)
Y_R2L_pred=svm_r2l.predict(X_R2L_test)
Y_U2R_pred=svm_u2r.predict(X_U2R_test)
     

# Create confusion matrix for DoS category
confusion_matrix_dos = pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
print("Confusion Matrix - DoS:")
print(confusion_matrix_dos)
print()

# Create confusion matrix for Probe category
confusion_matrix_probe = pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
print("Confusion Matrix - Probe:")
print(confusion_matrix_probe)
print()

# Create confusion matrix for R2L category
confusion_matrix_r2l = pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
print("Confusion Matrix - R2L:")
print(confusion_matrix_r2l)
print()

# Create confusion matrix for U2R category
confusion_matrix_u2r = pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
print("Confusion Matrix - U2R:")
print(confusion_matrix_u2r)
print()  



from sklearn.model_selection import cross_val_score
from sklearn import metrics

print("DoS Validation:")
accuracy = cross_val_score(svm_dos, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(svm_dos, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(svm_dos, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(svm_dos, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
print()  

print("DoS Validation:")
accuracy = cross_val_score(svm_probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(svm_probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(svm_probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(svm_probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')
print("F-mesaure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
print()

print("R2L Validation:")
accuracy = cross_val_score(svm_r2l, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(svm_r2l, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(svm_r2l, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(svm_r2l, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
print()

print("U2R Validation:")
accuracy = cross_val_score(svm_u2r, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(svm_u2r, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(svm_u2r, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(svm_u2r, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
print()