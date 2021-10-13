import numpy as np
import pytest
from jina import Document, DocumentArray
from pn_encoder import PNEncoder


@pytest.fixture
def docs():
    return DocumentArray([Document(blob=np.random.rand(2048, 3))])


def test_empty_docs(default_encoder):
    da = DocumentArray()
    default_encoder.encode(da)
    assert len(da) == 0


def test_encode_docs(default_encoder, docs):
    default_encoder.encode(docs)
    for doc in docs:
        assert doc.embedding is not None
        assert doc.embedding.shape == (128,)
