FROM jinaai/jina:2.0.23-py37-standard

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY ./ /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]