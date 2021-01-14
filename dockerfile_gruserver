# Start from the miniconda image
FROM dangawne/ubuntu_miniconda3x

# Copy the source code into the docker container
COPY ./model_api /model_api

# Create the conda environment
WORKDIR /model_api
RUN conda env create --file ./env_flask.yml
RUN conda env update --name mlflow-env --file ./model/conda.yaml

# Activate the environment and run the server
# This is a bit tricky. See: https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker
ENTRYPOINT ["conda", "run", "-n", "mlflow-env", "python", "./flask_api.py"]

# To build, type docker build -f ./dockerfile_gruserver -t dangawne/energy_gru .
# docker run -p 6000:6000 dangawne/energy_gru
