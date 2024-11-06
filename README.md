# Lead-Data-Scientist-Test

This repository contains tools and resources for analyzing call transcripts and predicting mental health.

## Setup

1. **Install Dependencies**  
   Ensure you have all necessary libraries installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**  
   Download the transcript data and place it in the workspace. After downloading, you should have a folder named `transcripts_v3` containing all the transcript `.txt` files.

3. **Label Data**  
   Labels for the last 20 transcripts are provided in this repository under the folder `transcripts_v3_labels`.
   
4. **OpenAI key**

   OpenAI key must be provided as an environment variable for the GenAI notebook to run. 

## Use Cases

### Use Case 1: Leveraging Generative AI for Call Transcript Analysis
- The notebook `GenAi_task.ipynb` was used for this task.
- Prompts for classification are stored in the Python file `prompts.py`.

### Use Case 2: Predicting Mental Health
- This use case utilizes three notebooks and one Python file:

  - `mental_health_EDA.ipynb`: Contains the exploratory data analysis (EDA) for understanding mental health data.
  - `mental_health_classic_ML.ipynb`: Performs experiments and evaluations for classic machine learning models.
  - `custom_network.py`: Implements a custom deep learning model using PyTorch and PyTorch Lightning.
  - `mental_health_DL.ipynb`: Trains and tests the custom deep learning model.

## Presentation

For a comprehensive overview of approaches, experimental procedures, and results, please refer to the presentation: [Presentation Link](<https://gamma.app/docs/Lead-Data-Science-Task-for-AXA-u16lm3q28rnirl6>)
