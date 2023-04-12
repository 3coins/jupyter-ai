import os

from jupyter_core.paths import jupyter_data_dir
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader


class DocumentIndexManager:
    _index_save_path = None
    _index = None

    @property
    def index_save_path(self):
        if not self._index_save_path:
            self._index_save_path = os.path.join(jupyter_data_dir(), "jupyter_ai_doc_index.json")

        return self._index_save_path

    @property
    def index(self):
        return self.index

    def __init__(self):
        try:
            self.index = GPTSimpleVectorIndex.load_from_disk(self.index_save_path)
        except Exception as e:
            self.index = GPTSimpleVectorIndex.load_from_string("This is just a dummy string to initialize the index.")

    def add_to_index(self, path: str, recreate: False):
        """Adds text documents at a path to the index"""

        documents = SimpleDirectoryReader(path)
        for document in documents:
            self.index.insert(document)
    
    def save_to_disk(self):
        """Saves the index to the disk"""
        self.index.save_to_disk(self.index_save_path)