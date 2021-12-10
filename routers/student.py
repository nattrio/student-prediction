from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient
from fastapi.responses import JSONResponse
from bson.objectid import ObjectId
from typing import List
from model.models import StudentModel, UpdateStudentModel
import os

router = APIRouter(
    prefix="/student",
    tags=['Student'],
    responses={404: {
        'message': "Not found!"
    }
    }
)

MONGODB_URL = os.environ["MONGODB_URL"]
client = MongoClient(MONGODB_URL)
db = client.dataset


@router.post("/", response_description="Add new student", response_model=StudentModel)
async def create_student(student: StudentModel = Body(...)):
    student = jsonable_encoder(student)
    new_student = db["students"].insert_one(student)
    created_student = db["students"].find_one({"_id": new_student.inserted_id})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_student)


@router.get(
    "/", response_description="List all students", response_model=List[StudentModel])
async def list_students():
    students = list(db["students"].find())
    return students


@router.get(
    "/{id}", response_description="Get a single student", response_model=StudentModel
)
async def show_student(id: str):
    if (student := db["students"].find_one({"_id": ObjectId(id)})) is not None:
        return student

    raise HTTPException(status_code=404, detail=f"Student {id} not found")


@router.put("/{id}", response_description="Update a student", response_model=StudentModel)
def update_student(id: str, student: UpdateStudentModel = Body(...)):
    student_temp = db["students"].find_one({"_id": ObjectId(id)})
    student = {k: v for k, v in student.dict().items() if v is not None}
    if len(student) < 1:
        return student_temp
    if student_temp:
        updated_student = db["students"].update_one(
            {"_id": ObjectId(id)}, {"$set": student}
        )
        if (
            updated_student := db["students"].find_one({"_id": ObjectId(id)})
        ) is not None:
            return updated_student

    if (existing_student := db["students"].find_one({"_id": ObjectId(id)})) is not None:
        print(existing_student)
        return existing_student
    
    raise HTTPException(status_code=404, detail=f"Student {id} not found")


@router.delete("/{id}", response_description="Delete a student")
async def delete_student(id: str):
    delete_result = db["students"].delete_one({"_id": ObjectId(id)})

    if delete_result.deleted_count == 1:
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)

    raise HTTPException(status_code=404, detail=f"Student {id} not found")
