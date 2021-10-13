import numpy as np
from jina import Flow, DocumentArray, Document


def test_flow():
    docs = DocumentArray([Document(blob=np.random.rand(2048, 3)) for _ in range(3)])
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(
            on='/index', inputs=docs, return_results=True
        )
        for doc in data[0].docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (128,)
