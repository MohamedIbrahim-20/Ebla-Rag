from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
from models import ResponseStatus
import os
import aiofiles
import logging
from .schemes.data import ProcessRequest


logger = logging.getLogger('uvicorn.error')
data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str,file: UploadFile, app_settings: Settings = Depends(get_settings)):
    data_controller = DataController()
    result = data_controller.validate_uploaded_file(file)
    if not result[0]:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": result[1]})
    
    file_path, file_id = data_controller.generate_unique_filename(project_id=project_id, original_filename=file.filename)
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await out_file.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, 
                            content={"message": ResponseStatus.FILE_UPLOAD_FAILED.value})
    return JSONResponse(status_code=status.HTTP_200_OK, 
                        content={"message": ResponseStatus.FILE_UPLOAD_SUCCESS.value, "file_id": file_id,})
    
    
@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, request: ProcessRequest):
    
    file_id = request.file_id
    chunk_size = request.chunk_size
    overlap_size = request.overlap_size
    process_controller = ProcessController(project_id=project_id)
    file_content = process_controller.get_file_content(filename=file_id)
    file_chunks = process_controller.process_file_content(file_content=file_content, chunk_size=chunk_size, chunk_overlap=overlap_size)
    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": ResponseStatus.FILE_PROCESSING_FAILED.value})
    # return JSONResponse(status_code=status.HTTP_200_OK, 
    #                     content={"message": ResponseStatus.PROCESSING_SUCCESS.value,
    #                                                             "chunks": file_chunks})
    return file_chunks