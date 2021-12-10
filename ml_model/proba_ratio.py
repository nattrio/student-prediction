
# Modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

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

group_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
   
def funcClass(score, scale_list=group_list):
        for i in range(len(scale_list)):
            if i == len(scale_list)-1:
                return len(scale_list)
            elif score < scale_list[i]:
                return i+1

def proba_group_kfold(Selected_model,X,y_class,r_state,focus_class=0):
    cv = StratifiedKFold(5,random_state=r_state,shuffle=True)
    
    over_df = pd.DataFrame(columns=["X","proba","group","y_test","y_pred"])

    
    for train_index, test_index in cv.split(X,y_class):
        X_train_k, y_train_k = X[train_index], y_class[train_index]
        X_test_k, y_test_k = X[test_index], y_class[test_index]
        
        Selected_model.fit(X_train_k, y_train_k)
        
        group = []
        y_pred = []
        proba_list = []

        for xi in X_test_k:
            proba = Selected_model.predict_proba([xi])
            proba_list.append(proba)
            group.append(funcClass(proba[0][focus_class]))
            y_pred.append(Selected_model.predict([xi])[0])
            
            # print(Selected_model.predict([xi]))

        test_zip = list(zip(X_test_k,proba_list,group,y_test_k,y_pred))
        test_df = pd.DataFrame(test_zip,columns=["X","proba","group","y_test","y_pred"])
        # print(test_df.groupby('group').size().reset_index(name='counts'))
        
        over_df = over_df.append(test_df,ignore_index=True)
     
    over_df = over_df.sort_values(by=['group'])
    over_df.index = range(len(over_df))   
    return over_df

def proba_cal(test_df):
    group_unique = sorted(test_df["group"].unique())

    # Loop all group
    result = pd.DataFrame()

    for g in group_unique:  
        temp_df = test_df[test_df["group"] == g]
        
        sample_0 = 0
        sample_1 = 0
        
        y_test_list = temp_df["y_test"].tolist()
        y_pred_list = temp_df["y_pred"].tolist()
          
        sample_size = temp_df.shape[0]
        
        for i in range(len(y_test_list)):
            if y_test_list[i] == 1:
                sample_0 += 1
            elif y_test_list[i] == 2:
                sample_1 += 1
                
        proba_sample_0 = sample_0/sample_size
        proba_sample_1 = sample_1/sample_size
        
        df_temp = pd.DataFrame([[g, sample_size, sample_0, sample_1, proba_sample_0, proba_sample_1]], 
                            columns=["group","sample",'sample_0','sample_1',"proba_0","proba_1"])
        
        test_df = test_df.sort_values(by=['group'])
        test_df.index = range(len(test_df))
        
        result = result.append(df_temp, ignore_index=True)

    return result

def proba_pipe(Selected_model,X,y_class,n_round,focus_class=0,):
    
    all_random_df = pd.DataFrame(columns=["X","proba","group","y_test","y_pred"])
    
    for r_state in range(n_round):
        test_df = proba_group_kfold(Selected_model,X,y_class,r_state,focus_class=0,)
        all_random_df = all_random_df.append(test_df)
        
        all_random_df = all_random_df.sort_values(by=['group'])
        all_random_df.index = range(len(all_random_df))
        
    result_df = proba_cal(all_random_df)
    return all_random_df, result_df


def fitting_proba(df,MODEL,OUT_SCALE):

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
    
    test_df,result_df = proba_pipe(Selected_model=selected_model,
                          X=X,
                          y_class=y_class,
                          n_round=3,
                          focus_class=0)

    Xg = result_df['group'].values.reshape(-1, 1)
    y0 = result_df['proba_0'].values.reshape(-1, 1)
    y1 = result_df['proba_1'].values.reshape(-1, 1)
    
    LR0 = LinearRegression()
    LR1 = LinearRegression()
    
    LR0.fit(Xg,y0)
    LR1.fit(Xg,y1)
    
    y0_pred = LR0.predict(Xg)
    y1_pred = LR1.predict(Xg)

    return LR0, LR1
    