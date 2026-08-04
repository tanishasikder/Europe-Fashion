import dagshub
import mlflow
import os

class Dagshub_Track():
    def __init__(self):
        self.owner = os.getenv('DAGSHUB_OWNER')
        self.repo = os.getenv('DAGSHUB_REPO')
        if not self.owner or not self.repo:
            raise ValueError("DAGSHUB_OWNER and DAGSHUB_REPO must be set")
        self._initialize = False

    def initialize(self):
        if not self._initialize:
            dagshub.init(repo_owner=self.owner, repo_name=self.repo, mlflow=True)
            self._initialize = True
