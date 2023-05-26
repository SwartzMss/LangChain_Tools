#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: search.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import os

from duckduckgo_search import ddg
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class SourceService(object):
    def __init__(self, config):
        self.vector_store = None
        self.config = config
        print(f"2. start load embedding  model: {self.config.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.docs_path = self.config.docs_path
        self.vector_store_path = self.config.vector_store_path

    def init_source_vector(self, file_list = []):
        """
        初始化本地知识库向量
        :return:
        """
        docs = []
        print(f"3. start searing file on local path: {self.config.docs_path}")
        index = 1
        for docName in file_list:
            print(f"   3.{index} : processing {docName}")
            index = index + 1
            if docName.endswith('.txt'):
                loader = UnstructuredFileLoader(f'{self.docs_path}/{docName}', mode="elements")
                doc = loader.load()
                docs.extend(doc)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"4. conguraulations ,service would start normally")

    def add_document(self, document_path):
        print(f"x. processing {document_path}")
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)

    # SYF: 暂时还没用到,无法校验文本是否变化，所以目前启动的时候都会重新生成       
    # cache/index.faiss  cache/index.pkl 这两个是缓存文件
    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store

