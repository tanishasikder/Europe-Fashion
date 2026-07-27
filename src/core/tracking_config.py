import dagshub
import mlflow
import os
from dotenv import load_dotenv

OWNER = os.getenv('DAGSHUB_OWNER')
REPO = os.getenv('DAGSHUB_REPO')

class Dagshub_Track():
    def __init__(self):
        self.owner = OWNER
        self.repo = REPO
        self._initialize = False

    def initialize(self):
        if not self._initialize:
            dagshub.init(repo_owner=self.owner, repo_name=self.repo, mlflow=True)
            self._initialize = True
