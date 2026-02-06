# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:25:19 2025

@author: 91788
"""

# import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# import file using pandas

temporary_df = pd.read_excel("C:/Users/91788/Desktop/Python/Data Science/Decision_tree/Projects/1.Heart_Disease/heart_disease.xlsx",
                             sheet_name="weight_of_evidence")


# removed IDs and target for better performance during model learing

target_removed = temporary_df.drop("Disease",axis = 1)
final_DF = target_removed.drop("customerid",axis = 1)

# creating another target DataFrame to give input in model
 
target = temporary_df["Disease"]


# using train_test_split for spilting data into training and testing with the help of sklearn.model_selection -> train_test_split()

sample_training , sample_testing , target_training , target_testing = train_test_split(
                                final_DF,target,random_state=42,
                                test_size=0.3,stratify=target)


# classifing Decision_Tree based on requirments(Hyperparameters)

decision_tree = DecisionTreeClassifier(
                min_samples_leaf=30,
               min_samples_split=20,
                max_depth=5,
                criterion="gini")


# training decision tree with :- .fit() , learns patters

decision_tree.fit(sample_training,target_training)


    # prediction on sample which left for testing , bcoz model knows the patter how it works

testing_prediction = decision_tree.predict(sample_testing)


# plotting decision with the help of sklearn.tree -> plot_tree

plt.figure(figsize=(30,15))

plot_tree(decision_tree,
          feature_names = final_DF.columns,
          class_names = ["No","Yes"],
          filled = True,
          rounded= True
          )

plt.title("Decision Tree For Heart Disease" , fontsize = 30)
plt.show()


# Finding confusion matrix of our model , how good/bad model runs...

print("Confusion Metrix : \n",confusion_matrix(target_testing , testing_prediction))
print("Classification Report : \n",classification_report(target_testing,testing_prediction))
print("Accuracy Score : ",accuracy_score(target_testing, testing_prediction))


# Finding most important feature in our model

important_DF = pd.DataFrame({
              "Features" : final_DF.columns,
              "Predicted_Values" : decision_tree.feature_importances_}).sort_values(by="Predicted_Values",ascending=False)

print(important_DF)


#importing important feature in excel file 

important_DF.to_excel("important_feature.xlsx")

#same for confusion matrix
confusion_DF = pd.DataFrame({
    "customer_IDs" : sample_testing.index,
    "Actual_Values" : target_testing.values,
    "Prediction" : testing_prediction})

confusion_DF.to_excel("Confusion_matrix.xlsx")










