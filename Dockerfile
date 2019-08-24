FROM conda/miniconda3:latest

WORKDIR /home

ADD env-kaggle.yaml /home

SHELL ["/bin/bash", "-c"]

# Update conda (otherwise will get incompatibility errors)
# Create environment from yaml file
# Change permissions so can run jupyter as non-root user
RUN conda update -n base -c defaults conda && \
    conda env create -f env-kaggle.yaml && \
    chmod -R 777 /home && \
    chmod -R 777 /usr/local/envs

ENV HOME /home

CMD /bin/bash -c "source activate kaggle && pip install -e . && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"

# Run the following commands:
# docker build -t kaggle .
# docker run -it -u `id -u` -v $(pwd):/home/kaggle -p 8888:8888 kaggle
