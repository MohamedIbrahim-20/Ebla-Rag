from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController
import os
data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str,file: UploadFile, app_settings: Settings = Depends(get_settings)):
    result = DataController().validate_uploaded_file(file)
    
    if not result[0]:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": result[1]})
    
    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_location = os.path.join(project_dir_path, file.filename)
    return {"signal": result[1]}