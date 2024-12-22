# Use an official Ubuntu 22.04 image as the base
FROM ubuntu:22.04

# Avoid prompts from apt (i.e. waiting user input to continue the install processes)
# Avoid prompts from apt and set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Update package lists and install system dependencies
# Removes the package lists to reduce the size of the final Docker image (for disk space & efficiency)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10


# Create a virtual environment
# It doesn't needed but for higher reproducibilty
RUN python3 -m venv venv

# Activate the virtual environment
ENV VIRTUAL_ENV=/venv \
    PATH="/venv/bin:$PATH"

# Set the working directory in the container 
WORKDIR /candidate_challenge

# Copy the requirements.txt file into the container 
COPY requirements.txt ./ 

# Install any needed packages specified in requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# Copies the rest of the files into the container (except of the .dockerignore)
COPY . ./

# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

# Define the program to execute when the container starts
ENTRYPOINT ["./entrypoint.sh"]


LABEL "creator"="Ioannis-Christos Kandias"
LABEL "purpose"="candidate challenge"
LABEL "description_1st_part"="Initial preprocessing removed columns with only NaNs but avoided aggressive cleaning to preserve dataset integrity. Data exploration considered source_id and quantity but was omitted as it deteriorated error quality. Models used included XGBoost, CatBoost, and LightGBM with MSE, MAE, and tuned Huber loss. Target scaling via log and quantile transformations addressed skewness. Cross-validation and Optuna-based hyperparameter tuning were applied. Final model used AutoGluon for efficiency, matching XGBoost performance."
LABEL "description_2nd_part" = "Data preprocessing included resizing images to RGB (128x128), tokenizing text with BERT, standardizing data, imputing missing values, and applying target transformations to address skewness. Model development used a CNN for images, BiLSTM for text, TabNet for tabular data, and modality fusion with MLP and MoE heads. Achieved MAE: 5689, Median Error: 498, MAPE: 18.7%."
