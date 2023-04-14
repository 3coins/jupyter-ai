import json
import os

from jupyter_core.paths import jupyter_data_dir
from llama_index import GPTEmptyIndex, GPTSimpleVectorIndex, SimpleDirectoryReader


class DocumentIndexManager:
    _index_save_path = None
    _indexed_dirs_save_path = None
    _doc_index = None
    root_dir = ""
    indexed_dirs = []

    @property
    def index_save_path(self):
        if not self._index_save_path:
            self._index_save_path = os.path.join(jupyter_data_dir(), "jupyter_ai_doc_index.json")

        return self._index_save_path
    
    @property
    def indexed_dirs_save_path(self):
        if not self._indexed_dirs_save_path:
            self._indexed_dirs_save_path = os.path.join(jupyter_data_dir(), "jupyter_ai_indexed_dirs.json")

        return self._indexed_dirs_save_path

    @property
    def doc_index(self):
        return self._doc_index

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        if os.path.exists(self.index_save_path):
            self._doc_index = GPTSimpleVectorIndex.load_from_disk(self.index_save_path)
            with open(self.indexed_dirs_save_path, 'r:utf-8') as fp:
                self.indexed_dirs = json.load(fp)
            print(f"Loaded index for dirs: {self.indexed_dirs}")
        else:
            self._doc_index = GPTEmptyIndex()
        
    def add_to_index(self, path: str, recreate: bool = False):
        """Adds text documents at a path to the index,
        path is relative to the jupyter server root dir.
        """

        documents = SimpleDirectoryReader(os.path.join(self.root_dir, path)).load_data()
        if isinstance(self.doc_index, GPTEmptyIndex):
            self._doc_index = GPTSimpleVectorIndex.from_documents(documents)
        else:
            for document in documents:
                self.doc_index.insert(document)
    
    def save_to_disk(self):
        """Saves the index to the disk"""
        self.doc_index.save_to_disk(self.index_save_path)
        with open(self.indexed_dirs_save_path, 'w') as f:
            f.write(json.dumps(self.indexed_dirs))
