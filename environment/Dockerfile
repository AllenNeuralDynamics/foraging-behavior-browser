# hash:sha256:51bda5f40316acb89ad85a82e996448f5a31d6f40b5b443e817e9b346eee2f67
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aiobotocore==2.4.0 \
    aiohttp==3.8.3 \
    aioitertools==0.11.0 \
    aiosignal==1.3.1 \
    altair==4.2.0 \
    async-timeout==4.0.2 \
    attrs==22.1.0 \
    blinker==1.5 \
    botocore==1.27.59 \
    cachetools==5.2.0 \
    certifi \
    charset-normalizer==2.1.1 \
    click==8.1.3 \
    commonmark==0.9.1 \
    contourpy==1.0.6 \
    cycler==0.11.0 \
    decorator==5.1.1 \
    dill==0.3.6 \
    distlib==0.3.4 \
    entrypoints==0.4 \
    extra-streamlit-components==0.1.56 \
    fonttools==4.38.0 \
    frozenlist==1.3.3 \
    fsspec==2022.11.0 \
    gitdb==4.0.9 \
    GitPython==3.1.29 \
    h5py==3.8.0 \
    hdmf==3.5.2 \
    idna==3.4 \
    importlib-metadata==5.0.0 \
    Jinja2==3.1.2 \
    jmespath==1.0.1 \
    jsonschema==4.17.1 \
    kiwisolver==1.4.4 \
    MarkupSafe==2.1.1 \
    matplotlib==3.6.2 \
    multidict==6.0.2 \
    nptyping==2.4.1 \
    numpy==1.23.5 \
    packaging==21.3 \
    pandas==1.5.2 \
    patsy==0.5.3 \
    Pillow==9.3.0 \
    pipenv==2022.5.2 \
    platformdirs==2.5.2 \
    plotly==5.11.0 \
    protobuf==3.20.3 \
    psrecord==1.2 \
    psutil==5.9.4 \
    pyarrow==10.0.1 \
    pydeck==0.8.0 \
    Pygments==2.13.0 \
    pyinstrument==4.1.1 \
    Pympler==1.0.1 \
    pynrrd==1.0.0 \
    pynwb==2.3.1 \
    pyparsing==3.0.9 \
    pyrsistent==0.19.2 \
    python-dateutil==2.8.2 \
    python-decouple==3.6 \
    pytz==2022.6 \
    pytz-deprecation-shim==0.1.0.post0 \
    requests==2.31.0 \
    rich==12.6.0 \
    ruamel.yaml==0.17.21 \
    ruamel.yaml.clib==0.2.7 \
    s3fs==2022.11.0 \
    scipy==1.10.0 \
    scikit-learn==1.3.2 \
    seaborn==0.11.2 \
    semver==2.13.0 \
    six==1.16.0 \
    smmap==5.0.0 \
    statannotations==0.5.0 \
    statsmodels==0.13.5 \
    streamlit==1.31.0 \
    streamlit-aggrid==0.3.5 \
    streamlit-nested-layout==0.1.1 \
    streamlit-plotly-events==0.0.6 \
    streamlit-profiler==0.2.4 \
    tenacity==8.1.0 \
    toml==0.10.2 \
    toolz==0.12.0 \
    tornado==6.2 \
    typing_extensions==4.9.0 \
    tzdata==2022.6 \
    tzlocal==4.2 \
    urllib3==1.26.13 \
    validators==0.20.0 \
    virtualenv==20.14.1 \
    virtualenv-clone==0.5.7 \
    watchdog==2.1.9 \
    wrapt==1.14.1 \
    yarl==1.8.1 \
    zipp==3.10.0 \
    git+https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training.git@main \
    pygwalker==0.4.7 \
    aind-data-access-api[docdb]==0.13.0 \
    streamlit-dynamic-filters==0.1.9 \
    semver==3.0.2


ADD "https://github.com/coder/code-server/releases/download/v4.21.1/code-server-4.21.1-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.21.1-linux-amd64/bin/code-server  /usr/bin/code-server


COPY postInstall /
RUN /postInstall
