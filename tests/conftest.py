import pytest
from pn_encoder import PNEncoder


@pytest.fixture(scope='session')
def default_encoder():
    return PNEncoder()
