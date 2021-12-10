
# Basic
import warnings
warnings.filterwarnings('ignore')

# Modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from ml_model.utils.preprocess import *
from ml_model.utils.modeling import *

def select_model(name):
    if name == 'MLP':
        selected_model = MLPClassifier(random_state=0)
        model_algo = 'MLPClassifier'
    elif name == 'RFC':
        selected_model = RandomForestClassifier(random_state=0)
        model_algo = 'RandomForestClassifier'
    elif name == 'KNN':
        selected_model = KNeighborsClassifier()
        model_algo = 'KNeighborsClassifier'
    elif name == 'LGR':
        selected_model = LogisticRegression(random_state=0)
        model_algo = 'LogisticRegression'
    elif name == 'DTC':
        selected_model = DecisionTreeClassifier(random_state=0)
        model_algo = 'DecisionTreeClassifier'
    return selected_model, model_algo

def trainig_model(df,MODEL,VAR_NUM,OUT_SCALE):
    
    VAR_NUM = ["School_GPAX", "Math", "Science", "English"]
    IN_CLASS = 5
    OUT_CLASS = 2
    IN_SCALE = 'score_range'
    
    # selected_model = MLPClassifier(random_state=0)
    selected_model, model_algo = select_model(MODEL)

    dataset = df
    droped_df = dropRecords(dataset,VAR_NUM)
    classified_df = toClass(droped_df, VAR_NUM, IN_CLASS, OUT_CLASS, IN_SCALE, OUT_SCALE)
    X,y,y_class = assignXY(classified_df, VAR_NUM)


    # mean of 5-kfold 3 round
    accuracy = kfold_score_k3(5,selected_model,X,y_class)
    
    selected_model.fit(X, y_class)
    
    return accuracy, selected_model, model_algo

def trainig_predict(df,MODEL,OUT_SCALE):

    # selected_model = MLPClassifier(random_state=0)
    VAR_NUM = ["School_GPAX", "Math", "Science", "English"]
    IN_CLASS = 5
    OUT_CLASS = 2
    IN_SCALE = 'score_range'
    
    selected_model, model_algo = select_model(MODEL)

    dataset = df
    droped_df = dropRecords(dataset,VAR_NUM)
    classified_df = toClass(droped_df, VAR_NUM, IN_CLASS, OUT_CLASS, IN_SCALE, OUT_SCALE)
    X,y,y_class = assignXY(classified_df, VAR_NUM)


    # mean of 5-kfold 3 round
    accuracy = kfold_score_k3(5,selected_model,X,y_class)
    
    selected_model.fit(X, y_class)
    
    return accuracy, selected_model, model_algo
    