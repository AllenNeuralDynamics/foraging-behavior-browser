# hash:sha256:51bda5f40316acb89ad85a82e996448f5a31d6f40b5b443e817e9b346eee2f67
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -r https://raw.githubusercontent.com/AllenNeuralDynamics/foraging-behavior-browser/main/requirements.txt --no-cache-dir

COPY postInstall /
RUN /postInstall
