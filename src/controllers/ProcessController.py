from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnums
class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        
    def get_file_extension(self, filename: str):
        return os.path.splitext(filename)[-1]
    
    def get_file_loader(self,  filename: str):
        file_path = os.path.join(self.project_path, filename)
        extension = self.get_file_extension(filename=filename).lower()
        if extension == ProcessingEnums.PDF.value:
            return PyMuPDFLoader(file_path)
        elif extension == ProcessingEnums.TXT.value:
            return TextLoader(file_path, encoding='utf-8')
        else:
            return None
    
    def get_file_content(self, filename: str):
        loader = self.get_file_loader(filename=filename)
        if loader:
            documents = loader.load()
            return documents
        return None
    
    def process_file_content(self, file_content: list, chunk_size: int = 100, chunk_overlap: int = 20):
        # documents = self.get_file_content(filename=filename)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        file_content_texts = [i.page_content for i in file_content]
        file_content_meta = [i.metadata for i in file_content] 
        chunks = text_splitter.create_documents(file_content_texts, metadatas=file_content_meta)
        return chunks
