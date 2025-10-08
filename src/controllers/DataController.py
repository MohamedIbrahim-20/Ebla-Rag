from fastapi import UploadFile
from .BaseController import BaseController
from .ProjectController import ProjectController
from models import ResponseStatus
import re
import os


class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1024 * 1024  # Convert MB to bytes


    def validate_uploaded_file(self, file: UploadFile):
        allowed_extensions = self.settings.FILE_ALLOWED_EXTENSIONS
        if file.content_type not in allowed_extensions:
            return False, ResponseStatus.FILE_TYPE_NOT_SUPPORTED.value
        if file.size > self.settings.MAX_FILE_SIZE  * self.size_scale:
            return False, ResponseStatus.FILE_SIZE_EXCEEDED.value
        return True, ResponseStatus.FILE_VALIDATED_SUCCESS.value
    
    def generate_unique_filename(self,project_id, original_filename: str) -> str:
        random_file_name = self.generate_random_string(12)
        project_path = ProjectController().get_project_path(project_id=project_id)
        cleaned_filename = self.get_clean_filename(original_filename)
        new_file_path = os.path.join(project_path, random_file_name + "_" + cleaned_filename)
        while os.path.exists(new_file_path):
            random_file_name = self.generate_random_string(12)
            new_file_path = os.path.join(project_path, random_file_name + "_" + cleaned_filename)
        return new_file_path, random_file_name + "_" + cleaned_filename
    
    def get_clean_filename(self, original_filename: str) -> str:
        # Remove any unwanted characters from the filename
        clean_name = re.sub(r'[^\w.]', '', original_filename.strip())
        clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
        return clean_name