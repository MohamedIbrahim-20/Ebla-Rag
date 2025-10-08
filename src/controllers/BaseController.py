from helpers.config import get_settings, Settings
import os
import random, string
class BaseController:
    def __init__(self):
        self.settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.files_dir = os.path.join(self.base_dir, "assets", "files")
        
    def generate_random_string(self, length=8):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choices(letters,k=length))
