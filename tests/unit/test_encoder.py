from typing import Tuple

import numpy as np
import pytest
from jina import Document, DocumentArray

from pn_encoder import PNEncoder


@pytest.fixture
def docs():
    return DocumentArray([Document(blob=np.random.rand(2048, 3)) for _ in range(3)])


@pytest.fixture(scope="function")
def nested_docs() -> DocumentArray:
    blob = np.random.rand(2048, 3)
    docs = DocumentArray([Document(id="root1", blob=blob)])
    docs[0].chunks = [
        Document(id="chunk11", blob=blob),
        Document(id="chunk12", blob=blob),
        Document(id="chunk13", blob=blob),
    ]
    docs[0].chunks[0].chunks = [
        Document(id="chunk111", blob=blob),
        Document(id="chunk112", blob=blob),
    ]

    return docs


def test_empty_docs(default_encoder):
    da = DocumentArray()
    default_encoder.encode(da)
    assert len(da) == 0


def test_none_docs(default_encoder):
    default_encoder.encode(docs=None, parameters={})


def test_encode_docs(default_encoder, docs):
    default_encoder.encode(docs)
    for doc in docs:
        assert doc.embedding is not None
        assert doc.embedding.shape == (128,)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(default_encoder, batch_size):
    docs = DocumentArray([Document(blob=np.random.rand(2048, 3)) for _ in range(32)])
    default_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (128,)


@pytest.mark.parametrize(
    "traversal_paths, counts",
    [
        [('c',), (('r', 0), ('c', 3), ('cc', 0))],
        [('cc',), (('r', 0), ('c', 0), ('cc', 2))],
        [('r',), (('r', 1), ('c', 0), ('cc', 0))],
        [('cc', 'r'), (('r', 1), ('c', 0), ('cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: Tuple[str],
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    default_encoder,
):
    default_encoder.encode(nested_docs, parameters={"traversal_paths": traversal_paths})
    for path, count in counts:
        embeddings = nested_docs.traverse_flat([path]).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count
