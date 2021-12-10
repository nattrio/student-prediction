from pydantic import BaseModel, Field
from typing import Optional, Union
from bson.objectid import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class StudentModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    Year: Optional[int] = Field(...)
    Program: Optional[str] = Field(...)
    Round: Optional[str] = Field(...)
    School_GPAX: Union[float, str] = Field(..., le=4.0)
    Math: Union[float, str] = Field(..., le=4.0)
    Science: Union[float, str] = Field(..., le=4.0)
    English: Union[float, str] = Field(..., le=4.0)
    SIT_GPAX: Union[float, str] = Field(..., le=4.0)
    School: Optional[str] = Field(...)
    Province: Optional[str] = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                'Year': 61,
                'Program': 'BScIT',
                'Round': 'เรียนดี',
                'School_GPAX': 3.13,
                'Math': 3.22,
                'Science': 2.97,
                'English': 2.75,
                'SIT_GPAX': 2.2,
                'School': 'S101-0',
                'Province': 'กรุงเทพมหานคร'
            }
        }


class UpdateStudentModel(BaseModel):
    Year: Optional[int]
    Program: Optional[str]
    Round: Optional[str]
    School_GPAX: Optional[Union[float, str]]
    Math: Optional[Union[float, str]]
    Science: Optional[Union[float, str]]
    English: Optional[Union[float, str]]
    SIT_GPAX: Optional[Union[float, str]]
    School: Optional[str]
    Province: Optional[str]
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                'Year': 61,
                'Program': 'BScIT',
                'Round': 'เรียนดี',
                'School_GPAX': 3.13,
                'Math': 3.22,
                'Science': 2.97,
                'English': 2.75,
                'SIT_GPAX': 2.2,
                'School': 'S101-0',
                'Province': 'กรุงเทพมหานคร'
            }
        }


class Subject(BaseModel):
    school_gpax: float
    math: float
    science: float
    english: float

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                'school_gpax': 2.5,
                'math': 2.5,
                'science': 2.5,
                'english': 2.5,
            }
        }
        
class SubjectExtra(BaseModel):
    school_gpax: float
    math: float
    science: float
    english: float
    model: str
    cutpoint: float

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                'school_gpax': 3,
                'math': 3,
                'science': 3,
                'english': 3,
                'model': "MLP",
                'cutpoint': 2.5
            }
        }


class ML_Model(BaseModel):
    var_num: list
    model: str
    cutpoint: float

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                'var_num': ["School_GPAX", "Math", "Science", "English"],
                'model': "MLP",
                'cutpoint': 2.5,
            }
        }
