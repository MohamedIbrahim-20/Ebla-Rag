from fastapi import UploadFile
from .BaseController import BaseController
from models import ResponseStatus
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