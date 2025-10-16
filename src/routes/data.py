from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, GroqRAGController
from models import ResponseStatus
import os
import aiofiles
import logging
from .schemes.data import ProcessRequest, IndexRequest, SearchRequest, AskRequest


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


# RAG Endpoints
@data_router.post("/index/{project_id}")
async def index_documents(project_id: str, request: IndexRequest, app_settings: Settings = Depends(get_settings)):
    """Index documents from CSV file for RAG"""
    try:
        rag_controller = GroqRAGController(project_id=project_id, settings=app_settings)
        # Initialize models
        if not rag_controller.initialize_models():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Failed to initialize models. Check GROQ_API_KEY."}
            )
        
        # Index documents
        # print("test")
        success = rag_controller.index_documents(request.csv_file)
        
        if success:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "Documents indexed successfully",
                    "project_id": project_id,
                    "csv_file": request.csv_file
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Failed to index documents"}
            )
            
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@data_router.post("/search/{project_id}")
async def search_documents(project_id: str, request: SearchRequest, app_settings: Settings = Depends(get_settings)):
    """Search for relevant documents"""
    try:
        rag_controller = GroqRAGController(project_id=project_id, settings=app_settings)
        
        # Initialize models
        if not rag_controller.initialize_models():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Failed to initialize models. Check GROQ_API_KEY."}
            )
        
        # Check if index exists based on method
        if request.method == "langchain":
            if not os.path.exists(rag_controller.LANGCHAIN_DB_PATH):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "No LangChain index found. Please run /index endpoint first."}
                )
        elif request.method == "llamaindex":
            if not os.path.exists(rag_controller.LLAMAINDEX_DB_PATH):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "No LlamaIndex found. Please run /index endpoint first."}
                )
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid method. Use 'langchain' or 'llamaindex'"}
            )
        
        # Search documents
        results = rag_controller.search_documents(request.query, request.method)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "query": request.query,
                "method": request.method,
                "results": results,
                "total_results": len(results)
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@data_router.post("/ask/{project_id}")
async def ask_question(project_id: str, request: AskRequest, app_settings: Settings = Depends(get_settings)):
    """Ask a question using RAG (Retrieval-Augmented Generation)"""
    try:
        rag_controller = GroqRAGController(project_id=project_id, settings=app_settings)
        
        # Initialize models
        if not rag_controller.initialize_models():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Failed to initialize models. Check GROQ_API_KEY."}
            )
        
        # Check if index exists based on method
        if request.method == "langchain":
            if not os.path.exists(rag_controller.LANGCHAIN_DB_PATH):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "No LangChain index found. Please run /index endpoint first."}
                )
            result = rag_controller.ask_question_langchain(request.question)
        elif request.method == "llamaindex":
            if not os.path.exists(rag_controller.LLAMAINDEX_DB_PATH):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "No LlamaIndex found. Please run /index endpoint first."}
                )
            result = rag_controller.ask_question_llamaindex(request.question)
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid method. Use 'langchain' or 'llamaindex'"}
            )
        
        if "error" in result:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=result
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@data_router.get("/status/{project_id}")
async def get_rag_status(project_id: str, app_settings: Settings = Depends(get_settings)):
    """Get RAG system status"""
    try:
        rag_controller = GroqRAGController(project_id=project_id, settings=app_settings)
        status_info = rag_controller.get_system_status()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=status_info
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )