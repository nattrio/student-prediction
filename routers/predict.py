
from fastapi import APIRouter
from pymongo import MongoClient
from model.models import Subject, SubjectExtra
from ml_model.model_training import *
from ml_model.proba_ratio import *
import pandas as pd
import pickle
import os

router = APIRouter(
    prefix="/api/v1/predict",
    tags=['Predict'],
    responses={404: {
        'message': "Not found!"
    }
    }
)

MONGODB_URL = os.environ["MONGODB_URL"]
client = MongoClient(MONGODB_URL)
db = client.trained


def predict_result(model, data):
    received = data.dict()

    school_gpax = received['school_gpax']
    math = received['math']
    science = received['science']
    english = received['english']

    subjects = [[school_gpax, math, science, english]]

    pred_class = model.predict(subjects).tolist()[0]
    proba_class = model.predict_proba(subjects).tolist()[0]
    
    return pred_class, proba_class


# @router.post("/mlp")
# def predict_mlp(subject: Subject):
#     model_mlp = pickle.loads(db["ml_model"].find_one({'name': "mlp"})["data"])

#     return predict_result(model_mlp, subject)


# @router.post("/rfc")
# def predict_rfc(subject: Subject):
#     model_rfc = pickle.loads(db["ml_model"].find_one({'name': "rfc"})["data"])

#     return predict_result(model_rfc, subject)


@router.post("/probation")
def predict_probation(subject: Subject):
    model_probation = pickle.loads(
        db["ml_model"].find_one({'name': "LogisticRegression", "cutpoint": 2.5})["data"])
    
    pred_class, proba_class = predict_result(model_probation, subject)
    
    model_desc = db["ml_model"].find_one({'name': "LogisticRegression", "cutpoint": 2.5})

    result = {
        'input': subject,
        'model': {
            '_id': str(model_desc['_id']),
            'name': model_desc["name"],
            'accuracy': model_desc["accuracy"],
            'cutpoint': model_desc["cutpoint"],
            'upload_date': model_desc["upload_date"], 
        },
        'prediction': pred_class,
        'probability': proba_class,
    }
    
    return result


@router.post("/predict-cutpoint")
def predict_cutpoint(subject_extra: SubjectExtra):
    students = list(client.dataset["students"].find())
    df = pd.DataFrame.from_dict(students)
    received = subject_extra.dict()
    
    model = received["model"]
    cutpoint = received["cutpoint"]

    acc, trained_model, model_algo = trainig_predict(
        df=df,
        MODEL=model,
        OUT_SCALE=cutpoint
    )
    
    school_gpax = received['school_gpax']
    math = received['math']
    science = received['science']
    english = received['english']
    
    subjects = [[school_gpax, math, science, english]]
    
    pred_class = trained_model.predict(subjects).tolist()[0] - 1
    proba_class = trained_model.predict_proba(subjects).tolist()[0]
    
    fitting_0, fitting_1 = fitting_proba(df,model,cutpoint)
    
    if pred_class == 0:
        proba_ratio = fitting_0.predict([[proba_class[0] * 10]])
    else:
        proba_ratio = fitting_1.predict([[proba_class[0] * 10]])
        
    fitting_result = proba_ratio[0][0]
        
    
    result = {
        'input': subject_extra,
        'accuracy': {
            'mean': acc[0],
            'class_0': acc[1],
            'class_1': acc[2],
        },
        'algorithm': model_algo,
        'prediction': pred_class,
        'probability': proba_class,
        'proba_ratio': fitting_result

    }
    return result