# Semantic_annotation_pipeline-extraction_module

# Short Description
This module is part of a semantic annotation pipeline, that handles the extraction of the information from a document.
# Installation
Two types of usage are available. One way is to use the dockerized version of this module. 
To use the dockerized version, see Installation with docker. 
The other way is to install it locally by executing the steps from Installation in environment. 


## Installation and execution with Python

### Install Anaconda with Python
```commandline
https://www.anaconda.com/products/individual 
```

### Create a new conda-Environment and install the environmental.yml
```
conda create -n NEW_ENVIRONMENT_NAME -f environmental.yml
conda activate NEW_ENVIRONMENT_NAME
```

### Install the requirements.txt
`
pip install -r requirements.txt
`
### Download the trained models
https://faubox.rrze.uni-erlangen.de/getlink/fiB1CqZjVGbadQo4MgX7KacR/Models.zip

This file includes the trained models for each subtask. Download it and add them to the specified folders. For the analysis-module add the models of the folder qa_models to: /core/apis/internal/QA_Pipeline/models/[TABLE_MODEL/TEXT_MODEL]
so it will be: 
- core/detection_models/models/image_model/img_detection_model.pth
- core/detection_models/models/table_model/gpu_table_detection_model.pth/cpu_table_detection_model.pth

### Download and install the other modules, to execute the annotation process.
https://github.com/ReneDorsch/semantic_annotation_pipeline-ui/
https://github.com/ReneDorsch/semantic_annotation_pipeline-annotation_module/
https://github.com/ReneDorsch/semantic_annotation_pipeline-analysis_module/


### Start the program
`python main.py`

## Installation with docker

### Install Docker
Install Docker with the following link:
```commandline
https://docs.docker.com/get-docker/
``` 
### Download the file
LINK TO FILE
