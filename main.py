import uvicorn
from fastapi import FastAPI
from routers import predict,model,student
from fastapi.middleware.cors import CORSMiddleware
import os

# load environment variables
port = int(os.environ["PORT"])


tags_metadata = [
    {
        "name": "default",
        "description": "This is **default** tags",
    },
    {
        "name": "Predict",
        "description": "Predict student class",
    },
    {
        "name": "Model",
        "description": "CRUD Model",
    },
    {
        "name": "Student",
        "description": "Upload student",
    },
]

# initialize FastAPI
app = FastAPI(
    title="StudentPrediction",
    description="## ระบบทำนายเกรดนักเรียนสำหรับการคัดเลือก",
    version="1.0.0",
    # docs_url="/documentation",
    # redoc_url=None,
    # openapi_tags=tags_metadata
    )

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"data": "Welcome to prediction 1"}
  
def config_router():
    app.include_router(predict.router)
    app.include_router(model.router)
    app.include_router(student.router)
    
config_router()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)