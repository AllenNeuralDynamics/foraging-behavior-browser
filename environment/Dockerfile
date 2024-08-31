# hash:sha256:51bda5f40316acb89ad85a82e996448f5a31d6f40b5b443e817e9b346eee2f67
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    streamlit==1.31.0 \
    streamlit-aggrid==0.3.5 \
    streamlit-bokeh3-events==0.1.4 \
    streamlit_dynamic_filters==0.1.9 \
    streamlit-nested-layout==0.1.1 \
    streamlit-plotly-events==0.0.6 \
    pygwalker==0.4.7 \
    extra-streamlit-components==0.1.56 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.2 \
    seaborn==0.11.2 \
    scipy==1.13.1 \
    plotly==5.11.0 \
    statsmodels==0.13.5 \
    statannotations==0.5.0 \
    scikit-learn==1.5.1 \
    semver==3.0.2 \
    s3fs==2024.6.1 \
    pillow==10.4.0 \
    aind-data-access-api[docdb]==0.13.0 \
    git+https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training.git@main \
    git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models.git@develop \
    git+https://github.com/AllenNeuralDynamics/aind-behavior-gym.git@develop


ADD "https://github.com/coder/code-server/releases/download/v4.21.1/code-server-4.21.1-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.21.1-linux-amd64/bin/code-server  /usr/bin/code-server


COPY postInstall /
RUN /postInstall
