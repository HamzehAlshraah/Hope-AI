#Calling the libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

# read data
data=pd.read_csv(r"C:\Users\user\Desktop\SDK\Project\projectMidML\student_depression_dataset.csv")

# drop the column that are not useful
data=data.drop(["id","City"],axis=1)
data["Financial Stress"].replace("?",data["Financial Stress"].mode()[0],inplace=True)

#Data processing type object in label encodeing
le = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    data[column] = le.fit_transform(data[column])

# split data feature and target
x=data.drop("Depression",axis=1)
y=data["Depression"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)

# train model Logistic Regression
lg_model=LogisticRegression()
lg_model.fit(x_train,y_train)

# train Random Forest Classifier
rf_model=RandomForestClassifier(n_estimators=9,max_depth=5)
rf_model.fit(x_train,y_train)

# train Decision Tree Classifier
dt_model=DecisionTreeClassifier(max_depth=5)
dt_model.fit(x_train,y_train)


# train support vector classifier
svm_model=SVC()
svm_model.fit(x_train,y_train)

# predict in all model
y_pred_lg=lg_model.predict(x_test)
y_pred_rf=rf_model.predict(x_test)
y_pred_dt=dt_model.predict(x_test)
y_pred_svm=svm_model.predict(x_test)
# upload model
joblib.dump(lg_model,"logistic_regression.pkl")
joblib.dump(rf_model,"random_forset.plk")
joblib.dump(dt_model,"decision_tree_classifier.plk")
joblib.dump(svm_model,"support_vector_classifier.plk")

# metrics : accuracy_score , classification_report , confusion_matrix in all model
acc_lg=accuracy_score(y_test, y_pred_lg)
acc_rf=accuracy_score(y_test, y_pred_rf)
acc_dt=accuracy_score(y_test, y_pred_dt)
acc_svm=accuracy_score(y_test, y_pred_svm)

cr_lg=classification_report(y_test,y_pred_lg)
cr_rf=classification_report(y_test,y_pred_rf)
cr_dt=classification_report(y_test,y_pred_dt)
cr_svm=classification_report(y_test,y_pred_svm)

cm_lg=confusion_matrix(y_test,y_pred_lg)
cm_rf=confusion_matrix(y_test,y_pred_rf)
cm_dt=confusion_matrix(y_test,y_pred_dt)
cm_svm=confusion_matrix(y_test,y_pred_svm)

# acc=pd.DataFrame({"Accuracy model Logistic Regression":[acc_lg],
#                   "Accuracy model Random Foarst Classifer":[acc_rf],
#                   "Accuracy model Decision Tree Classifer": [acc_dt],
#                   "Accuracy model Support Vector Classifer":[acc_svm],
#                   "            ":" " ,
#                   "Classification report model Logistic Regression ":[cr_lg],
#                   "Classification report model Random Foarst Classifer":[cr_rf],
#                   "Classification report model Decision Tree Classifer":[cr_dt],
#                   "Classification report model Support Vector Classifer":[cr_svm]
#                  })
Accuracy = pd.DataFrame({
                        "Model": ["Logistic Regression",
                                  "Random Forest",
                                  "Decision Tree",
                                  "SVM"],
    
                       "Accuracy": [int(acc_lg*100),
                                    int(acc_rf*100),
                                    int(acc_dt*100),
                                    int(acc_svm*100)]})


classification_report=pd.DataFrame({
                                    "Model": ["Logistic Regression",
                                              "Random Forest",
                                              "Decision Tree",
                                              "SVM"],
                                    
                                    "Classification Report":[cr_lg,
                                                             cr_rf,
                                                             cr_dt,
                                                             cr_svm]})

confusion_matrix=pd.DataFrame({   
                                "Model": ["Logistic Regression",
                                          "Random Forest",  
                                          "Decision Tree",
                                          "SVM"],
                               
                               "Confusion Matrix":[cm_lg,
                                                   cm_rf,
                                                   cm_dt,
                                                   cm_svm],})
Accuracy.to_csv("Accuracy.csv",index=False)
classification_report.to_csv("classification_report.csv",index=False)
confusion_matrix.to_csv("confusion_matrix.csv",index=False)
