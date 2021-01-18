
 
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing




# read data and preprocessing
data_name='spambase.csv'
data_pandas = pandas.read_csv(data_name)
data_main=np.array(data_pandas)
lbl=data_main[:,-1]
data=data_main[:,:-1]
data = preprocessing.scale(data) 


# split data
x_train, x_test, y_train, y_test = train_test_split(data, lbl, test_size=0.20)


#SVM
SVM_clf= SVC(gamma='auto')
SVM_clf.fit(x_train, y_train)
pre = SVM_clf.predict(x_test)

classfi_report=classification_report(y_test, pre,output_dict=True)

# save out
out_array=[]
out_array.append(accuracy_score(y_test, pre))
out_array.append(classfi_report['macro avg']['precision'] )
out_array.append(classfi_report['macro avg']['recall'])
out_array.append( classfi_report['macro avg']['f1-score'])


# plot
fig=plt.figure(figsize=(6, 8)) 
plt.bar( ['Accuracy','Precision','Recall','F1-Score'],out_array,color = ['black', 'blue','red','grey'])
plt.xticks(['Accuracy','Precision','Recall','F1-Score'], rotation=80)
plt.ylabel('%')
plt.title(' SVM Classification on Spambase Dataset ')
for idx, value in enumerate(out_array):
   value=round(value,2)
   plt.text(idx-0.1 , value+0.01 , str(value), color='blue', fontweight='bold')

   plt.text(idx-0.1 , value+0.01 , str(value), color='blue', fontweight='bold')
fig.show()

