# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve , roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# load dataset
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# features and target variable
X = df[['max heart rate']]
y = df['target']

# data is split into training and testing sets
X_train , X_test , y_train , y_test = train_test_split (X, y, random_state = 42, test_size = 0.2, stratify = y)

# decision tree
clf = DecisionTreeClassifier (max_depth = 3, random_state = 42)
clf.fit(X_train , y_train)
final_y_pred = clf.predict(X_test)
y_prob_tree = clf.predict_proba(X_test)[:, 1]
fpr_tree , tpr_tree , thresholds_tree = roc_curve(y_test , y_prob_tree)
auc_tree = round(roc_auc_score(y_test , y_prob_tree) ,5)
print("Decision Tree AUC:", auc_tree)

# log reg model
log_reg = LogisticRegression (max_iter = 1000, random_state = 42)
log_reg.fit(X_train , y_train)

# predict probabilities (for ROC curve)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

# fpr , tpr , thresholds for log reg
fpr_log , tpr_log , thresholds_log = roc_curve(y_test , y_prob_log)
auc_log = round(roc_auc_score(y_test , y_prob_log), 5)
print("\nLogistic Regression AUC:", auc_log)

# random forest classifier
rf_clf = RandomForestClassifier (random_state = 42)
rf_clf.fit(X_train , y_train)

# predict probabilities for ROC
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

# fpr , tpr , thresholds for log reg
fpr_rf , tpr_rf , thresholds_rf = roc_curve(y_test , y_prob_rf)
auc_rf = round(roc_auc_score(y_test , y_prob_rf), 5)
print("\nRandom Forest AUC: ", auc_rf)

# plot ROC curves
plt.figure(figsize = (7, 7))
plt.plot(fpr_log , tpr_log , label = "Logistic Regression AUC: " + str( auc_log), color = "blue")
plt.plot(fpr_tree , tpr_tree , label = "Decision Tree AUC: " + str(auc_tree) , color = "green")
plt.plot(fpr_rf , tpr_rf , label = "Random Forest AUC: " + str(auc_rf), color = "red")
plt.plot ([0, 1], [0, 1], "k--", label = "Random Guess")

# mark every 5th point for log regression
for i in range (0, len(fpr_log), 5):
    plt.scatter(fpr_log[i], tpr_log[i], color = "blue")
    plt.text(fpr_log[i]+0.02 , tpr_log[i], str(round(thresholds_log [i], 5)) , color = "blue", fontsize = 8)

# mark every 5th point for decision tree
for i in range (0, len(fpr_tree), 5):
    plt.scatter(fpr_tree[i], tpr_tree[i], color = "green")
    plt.text(fpr_tree[i]+0.02 , tpr_tree[i], str(round(thresholds_tree [i], 5)), color = "green", fontsize = 8)

# mark every 5th point for Random Forest
for i in range (0, len(fpr_rf), 5):
    plt.scatter(fpr_rf[i], tpr_rf[i], color = "red")
    plt.text(fpr_rf[i]+0.02 , tpr_rf[i], str(round(thresholds_rf[i], 5)), color = "red", fontsize = 8)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Comparison: Cholesterol")
plt.legend(loc="lower right")
plt.grid(True)
plt.show ()

# logistic regression ROC points
roc_log = pd.DataFrame ({
    "FPR": fpr_log ,
    "TPR": tpr_log ,
    "Threshold": thresholds_log
}).round (5)
roc_log.to_csv("roc_logistic_regression.csv", index = False)

# decision tree ROC points
roc_tree = pd.DataFrame ({
    "FPR": fpr_tree ,
    "TPR": tpr_tree ,
    "Threshold": thresholds_tree
}).round (5)
roc_tree.to_csv("roc_decision_tree.csv", index = False)

# random first ROC points
roc_rf = pd.DataFrame ({
    "FPR": fpr_rf ,
    "TPR": tpr_rf ,
    "Threshold": thresholds_rf
}).round (5)
roc_rf.to_csv("roc_random_forest.csv", index = False)

# trapezoidal AUC calculation function
# choose fpr and tpr columns based on csv file headings
def trapezoid(data , fpr_col = "FPR", tpr_col = "TPR", sort = True):
    # reads in CSV file
    df = pd.read_csv(data)
    # removes missing values
    df = df[[ fpr_col , tpr_col ]]. dropna ()
    # ensures data is sorted from smallest to largest , so integral is computed from left to right
    if sort:
        # fpr represents x, so it ensures the x-values are sorted
        df = df.sort_values(by=fpr_col , inplace=False)
    # column values in CSV file are converted to arrays
    fpr = df[fpr_col ]. to_numpy ()
    tpr = df[tpr_col ]. to_numpy ()
    # new array of consecutive (2nd - 1st) fpr values subtracted from each other
    dx = fpr [1:] - fpr [:-1]
    # new array of averaged tpr values (2nd + 1st)
    avg = (tpr [1:] + tpr [: -1]) / 2.0
    # multiply elements -wise and add all
    auc = np.sum(dx * avg)
    return auc

# midpoint Riemann sum AUC calculation function
# choose fpr and tpr columns based on CSV file headings
def midpoint(data , fpr_col="FPR", tpr_col="TPR", sort=True):
    # reads in CSV file
    df = pd.read_csv(data)
    # removes missing values
    df = df[[ fpr_col , tpr_col ]]. dropna ()
    # ensures data is sorted from smallest to largest , so integral is computed from left to right
    if sort:
        # fpr represents x, so it ensures the x-values are sorted
        df = df.sort_values(by=fpr_col , inplace=False)
    # column values in CSV file are converted to arrays
    fpr = df[fpr_col ]. to_numpy ()
    tpr = df[tpr_col ]. to_numpy ()
    # new array of consecutive (2nd - 1st) fpr values subtracted from each other for widths
    dx = fpr [1:] - fpr [:-1]
    # boolean to determine if the width is zero to get rid of FPR values
    pos = dx > 0
    # create arrays with the start and end points for each pair of values for each column
    fpr_left = fpr [: -1][ pos]
    fpr_right = fpr [1:][ pos]
    tpr_left = tpr [: -1][ pos]
    tpr_right = tpr [1:][ pos]
    # array without zeros
    dx = dx[pos]
    # array of midpoint x-values
    mid_fpr = (fpr_left + fpr_right) / 2.0
    # slope of tpr/fpr
    slope = (tpr_right  - tpr_left) / dx
    # find midpoint tpr value based on midpoint fpr value
    mid_tpr = tpr_left + slope * (mid_fpr  - fpr_left)
    # multiply elements -wise and add all
    auc = np.sum(mid_tpr * dx)
    return auc

# Simpson ’s rule AUC calculation function
def simpsons(data , fpr_col="FPR", tpr_col="TPR", sort=True):
    # reads in CSV file
    df = pd.read_csv(data)
    # removes missing values
    df = df[[ fpr_col , tpr_col ]]. dropna ()
    # ensures data is sorted from smallest to largest , so integral is computed from left to right
    if sort:
        # fpr represents x, so it ensures the x-values are sorted
        df = df.sort_values(by=fpr_col , inplace=False)
    # column values in CSV file are converted to arrays
    fpr = df[fpr_col ]. to_numpy ()
    tpr = df[tpr_col ]. to_numpy ()
    # new array of consecutive (2nd - 1st) fpr values subtracted from each other for widths
    dx = fpr [1:] - fpr [:-1]
    pos = dx > 0
    fpr = fpr [: -1][ pos]. tolist () + [fpr [-1]]
    tpr = tpr [: -1][ pos]. tolist () + [tpr [-1]]
    n = len(fpr)
    auc = 0.0
    # simpson ’s rule on three consecutive points since the points aren ’t equally spaced
    i = 0
    while i < n - 2:
        # create temporary three points
        x0 , x1 , x2 = fpr[i], fpr[i+1], fpr[i+2]
        y0 , y1 , y2 = tpr[i], tpr[i+1], tpr[i+2]
        # distance between points
        dx = (x2 - x0) / 2
        # calculate the auc for these points
        auc += dx / 3 * (y0 + 4*y1 + y2)
        # move to the next triple
        i += 2

    # if there is one leftover interval at the end , calculate using trapezoid
    if i == n - 2:
        auc += (fpr[-1] - fpr [-2]) * (tpr[-2] + tpr [-1]) / 2
    return auc


# time it takes for each AUC calculation
start = round(time.time (), 5)
auc_log = round(trapezoid("roc_logistic_regression.csv") ,5)
end = round(time.time (), 5)
print("Logistic Regression AUC (Trapezoidal): " + str(auc_log) + " Time: " + str(round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_tree = round(trapezoid("roc_decision_tree.csv"), 5)
end = round(time.time (), 5)
print("Decision Tree AUC (Trapezoidal): " + str(auc_tree) + " Time: " + str(round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_rf = round(trapezoid("roc_random_forest.csv"), 5)
end = round(time.time (), 5)
print("Random Forest AUC (Trapezoidal): " + str(auc_rf) + " Time: " + str( round(end - start , 5)) +   " seconds")

# time it takes for each midpoint calculation
start = round(time.time (), 5)
auc_log = round(midpoint("roc_logistic_regression.csv") ,5)
end = round(time.time (), 5)
print("Logistic Regression AUC (Midpoint): " + str(auc_log) + " Time: " + str(round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_tree = round(midpoint("roc_decision_tree.csv"), 5)
end = round(time.time (), 5)
print("Decision Tree AUC (Midpoint): " + str(auc_tree) + " Time: " + str( round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_rf = round(midpoint("roc_random_forest.csv"), 5)
end = round(time.time (), 5)
print("Random Forest AUC (Midpoint): " + str(auc_rf) + " Time: " + str( round(end - start , 5)) +   " seconds")

# time it takes for each simpson ’s rule calculation
start = round(time.time (), 5)
auc_log = round(simpsons("roc_logistic_regression.csv") ,5)
end = round(time.time (), 5)
print("Logistic Regression AUC (Simpsons): " + str(auc_log) + " Time: " + str(round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_tree = round(simpsons("roc_decision_tree.csv"), 5)
end = round(time.time (), 5)
print("Decision Tree AUC (Simpsons): " + str(auc_tree) + " Time: " + str( round(end - start , 5)) +   " seconds")

start = round(time.time (), 5)
auc_rf = round(simpsons("roc_random_forest.csv"), 5)
end = round(time.time (), 5)
print("Random Forest AUC (Simpsons): " + str(auc_rf) + " Time: " + str( round(end - start , 5)) +   " seconds")
