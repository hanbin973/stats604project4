# Use the Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Create a conda environment and install all packages in one go.
# Added 'requests' for Python-based API calls.
RUN conda create -n pjm_env -c conda-forge \
    python=3.10 \
    statsmodels \
    numpy \
    pandas \
    scipy \
    requests \
    && conda clean -afy

# Activate the conda environment for all subsequent commands
SHELL ["conda", "run", "-n", "pjm_env", "/bin/bash", "-c"]

# --- Your Original Data Download ---
# Download the OSF data file
RUN wget -O data.zip "https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip="

# Create a directory to hold the unzipped data
RUN mkdir -p /app/data

# Un-zip the file, then extract the nested tar.gz file, then clean up archives
RUN echo "Unzipping main zip file..." \
    && unzip data.zip -d /app/data \
    && echo "Unzipping nested tar.gz file..." \
    && tar -xzf /app/data/hrl_load_metered_2016-2025.tar.gz -C /app/data \
    && echo "Cleaning up archive files..." \
    && rm data.zip \
    && rm /app/data/hrl_load_metered_2016-2025.tar.gz \
    && echo "Data extraction complete."
# --- End of Your Original Data ---


# --- Weather Download Removed ---
# We no longer download a static weather.json file here.
# The load_data.py script will now fetch this data dynamically.


# Copy the Python script into the container's working directory
COPY load_data.py .

# Set the default command to run when the container starts
# Use the "shell" form (no []) so it respects the SHELL directive above
CMD python load_data.py
