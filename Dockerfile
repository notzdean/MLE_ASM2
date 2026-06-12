FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow

# System dependencies: Java for PySpark, curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jdk-headless \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Symlink JAVA_HOME to a stable path that works on both amd64 and arm64
RUN ln -s "$(dirname $(dirname $(readlink -f $(which java))))" /usr/lib/jvm/java-home
ENV JAVA_HOME=/usr/lib/jvm/java-home
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Apache Airflow pinned to a single compatible set via constraints
ARG AIRFLOW_VERSION=2.9.3
ARG PYTHON_VERSION=3.12
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install ML and data libraries
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /opt/airflow
RUN mkdir -p dags logs plugins datamart model_store

EXPOSE 8080
