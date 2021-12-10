from fastapi import APIRouter, HTTPException, status
from pymongo import MongoClient
from fastapi.responses import JSONResponse
from bson.objectid import ObjectId
import pandas as pd
import pickle
import os
from datetime import datetime
from ml_model.model_training import *
from model.models import ML_Model

router = APIRouter(
    prefix="/model",
    tags=['Model'],
    responses={404: {
        'message': "Not found!"
    }
    }
)

MONGODB_URL = os.environ["MONGODB_URL"]
client = MongoClient(MONGODB_URL)
db = client.trained


@router.get("/")
async def show_model_list():
    model_dropdown = {
        "model_lists": [
            {
                "id": 0,
                "value": "MLP",
                "name": "MLPClassifier"
            },
            {
                "id": 1,
                "value": "RFC",
                "name": "RandomForestClassifier"
            },
            {
                "id": 2,
                "value": "LGR",
                "name": "LogisticRegression"
            },
            {
                "id": 3,
                "value": "DTC",
                "name": "DecisionTreeClassifier"
            },
            {
                "id": 4,
                "value": "KNN",
                "name": "KNeighborsClassifier"
            }
        ]
    }

    return model_dropdown


@router.get("/detail")
async def show_model_detail():
    models_list = list(db["ml_model"].find({}))
    models_info = []
    for m in models_list:
        temp = {
            '_id': str(m['_id']),
            'name': m["name"],
            'accuracy': m["accuracy"],
            'cutpoint': m["cutpoint"],
            'upload_date': m["upload_date"],
        }
        models_info.append(temp)
    return models_info


@router.post("/")
async def create_model(ml_model: ML_Model):
    students = list(client.dataset["students"].find())
    df = pd.DataFrame.from_dict(students)

    received = ml_model.dict()
    # print(received)
    cutpoint = received["cutpoint"]

    acc, trained_model, model_algo = trainig_model(df,
                                                   VAR_NUM=received["var_num"],
                                                   MODEL=received["model"],
                                                   OUT_SCALE=cutpoint
                                                   )

    # print(acc)
    # print(trained_model.predict([[4, 4, 4, 4]]))

    pickled_data = pickle.dumps(trained_model)
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_insert = {
        'name': model_algo,
        'cutpoint': cutpoint,
        'data': pickled_data,
        'accuracy': {
            'mean': acc[0],
            'class_0': acc[1],
            'class_1': acc[2],
        },
        'upload_date': upload_date
    }

    db["ml_model"].insert_one(model_insert)

    res = {
        'name': model_algo,
        'cutpoint': cutpoint,
        'accuracy': {
            'mean': acc[0],
            'class_0': acc[1],
            'class_1': acc[2],
        },
        'upload_date': upload_date
    }

    return res


@router.delete("/{id}", response_description="Delete a model")
def delete_model(id: str):
    delete_result = db["ml_model"].delete_one({"_id": ObjectId(id)})

    if delete_result.deleted_count == 1:
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)

    raise HTTPException(status_code=404, detail=f"Model {id} not found")
