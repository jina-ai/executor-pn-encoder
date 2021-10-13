from typing import Dict, Iterable

import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator


from pn import get_bottleneck_model


class PNEncoder(Executor):
    """
    PNEncoder encodes a Document containing point sets (point cloud) in its blob and adds embedding to it using the
    pointnet model. The embedding has shape (128,)
    """
    def __init__(
            self,
            ckpt_path: str = 'ckpt/ckpt_True',
            traversal_paths: Iterable[str] = ('r',),
            batch_size: int = 32,
            **kwargs):
        """
        :param ckpt_path: model checkpoint file path
        :param traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
        :param batch_size: size of each batch
        """
        super().__init__(**kwargs)
        self.embedding_model = get_bottleneck_model(ckpt_path=ckpt_path)
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.logger = JinaLogger('pn-encoder')

    @requests(on=['/index', '/search'])
    def encode(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        :param docs: a `DocumentArray` contains `Document`s with `blob` of the size (2048, 3).
            The `blob` contains point cloud data, obtained after sampling 2048 from the 3D object mesh.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`.
            For example, `parameters={'traversal_paths': ['r'], 'batch_size': 10}` will
            override the `self.traversal_paths` and `self.batch_size`.
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.traversal_paths),
                batch_size=parameters.get('batch_size', self.batch_size),
                needs_attr='blob',
            )
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            blob_batch = np.stack([d.blob for d in document_batch])
            embedding_batch = self.embedding_model.predict(blob_batch)[-1]
            for document, embedding in zip(document_batch, embedding_batch):
                document.embedding = embedding
