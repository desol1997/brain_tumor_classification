# Brain Tumor Classification

## About

This project is dedicated to developing a robust deep learning model for the classification of brain tumors using TensorFlow and Keras. The ultimate goal is to deploy this model on AWS Lambda, creating a scalable and efficient solution for classifying brain tumors based on MRI images. The model's accuracy and real-time capabilities make it a valuable tool for medical professionals seeking prompt and reliable brain tumor diagnoses.

## Table of Contents

- [About](#about)
- [Use Cases](#use-cases)
- [Data](#data)
- [Deep Learning Model](#deep-learning-model)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Deployment](#deployment)

## Use Cases

### Medical Diagnosis

- **On-Demand Image Classification:** The model deployed on AWS Lambda enables healthcare professionals to classify brain tumors in real-time by submitting MRI images. This on-demand capability supports quicker decision-making in critical medical situations.

### Telemedicine

- **Remote Diagnostics:** Facilitate remote diagnostics by allowing medical practitioners to securely submit MRI images for immediate brain tumor classification. This can enhance telemedicine services, providing timely insights into patients' conditions.

### Research and Education

- **Interactive Learning:** Researchers and educators can utilize the deployed model to create interactive learning experiences. Students can submit MRI images to observe the model's classification, fostering a deeper understanding of brain tumor characteristics.

### Mobile Health Applications

- **In-App Tumor Classification:** Integrate the model into mobile health applications, allowing users to assess brain tumor likelihood by capturing and submitting MRI images directly through their smartphones. This empowers individuals with quick and accessible preliminary information.

### Clinical Trials Screening

- **Efficient Participant Screening:** Streamline the screening process for clinical trials by employing the model to rapidly evaluate MRI images submitted by potential participants. This can accelerate the recruitment phase for brain tumor-related studies.

### Emergency Room Decision Support

- **Immediate Triage:** In emergency room scenarios, the model can serve as a decision support tool, aiding healthcare professionals in prioritizing cases by quickly assessing the presence of brain tumors in incoming patients through real-time image classification.

## Data

The model is trained on a comprehensive [dataset of MRI images](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data) available on Kaggle, ensuring its ability to accurately classify various types of brain tumors.

## Deep Learning Model

The neural network based on InceptionV3 architecture, implemented using TensorFlow and Keras, provides a working solution for brain tumor classification.

## Dependencies

This project uses `pipenv` for managing the Python environment and dependencies. Ensure you have the following installed:

- Python (version 3.10 or higher)
- `pipenv` (Python package for managing virtual environments)

## Installation

### Step 1: Python and Git

If not already installed, download and install Python from [Python's official website](https://www.python.org/).
Additionally, you'll need Git to clone the repository. Install Git from [Git's official website](https://git-scm.com/) if not already present.

### Step 2: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/desol1997/brain_tumor_classification.git
cd hotel_booking_cancellation_prediction
```

### Step 3: Set Up the Environment

Ensure `pipenv` is installed. If not, install it via pip:

```bash
pip install pipenv
```

Then, create the virtual environment and install dependencies using `pipenv`:

```bash
pipenv install
```

### Step 4: Activate the Environment

Activate the virtual environment to work within it:

```bash
pipenv shell
```

### Step 5: Running the Project

Once the environment is activated, follow specific instructions provided in the project's documentation or source code to run the project.

## Deployment

The deep learning model has been deployed as a web service using Docker and AWS Lambda. You can test the service either locally with Docker or deploy it to AWS Lambda and set up AWS API Gateway.

### Local Deployment with Docker

1. **Build the Docker Image:**
   - Use the following command to build the Docker image:
   ```bash
   docker build -t myapp .
   ```
   - Replace `myapp` with the name you want to assign to the Docker image.

2. **Run the Docker Container:**
   - Start a container locally using the following command:
   ```bash
   docker run -p 8080:8080 myapp
   ```

3. **Access the Application:**
   - Update the `url` variable in the `test.py` script located in the `src` folder with `http://localhost:8080/2015-03-31/functions/function/invocations`. After making this change, execute the `test.py` script. This will enable you to view the response in JSON format, displaying the result of brain tumor classification.

### Deployment to AWS Lambda

1. **AWS Account:**
   - Before deploying to AWS Lambda, ensure you have an active AWS account.

2. **Estabclish a Repository in AWS Elastic Container Registry:**
   - Refer to the [AWS documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html) for a detailed walkthrough on pushing a container image to a private Amazon ECR repository.

3. **Set up AWS Lambda Function:**
   - Follow the steps outlined in the [AWS documentation](https://docs.aws.amazon.com/lambda/) to create a Lambda function..
   - When configuring the Lambda function, opt for a Container image and select the image you've previously pushed to the AWS Elastic Container Registry.
   - Ensure that you set the timeout to a minimum of 30 seconds and allocate at least 1024MB of memory.

4. **Configure AWS API Gateway:**
   - Consult the [AWS documentation](https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html) for step-by-step instructions on setting up an API Gateway.
   - During the API Gateway configuration, opt for REST API as the API type.
   - Once the REST API is in place, create a `/predict` resource.
   - Subsequently, establish a POST method for the resource. This method will handle POST requests containing the image URL for classification.
   - Finally, deploy the configured API with the `test` Stage name.

5. **Access the Application:**
   - Update the `url` variable in the `test.py` script located in the `src` folder with the provided URL or endpoint from AWS API Gateway. After making this change, execute the `test.py` script. This will enable you to view the response in JSON format, displaying displaying the result of brain tumor classification.

The outcomes of the deployment are visible in the image stored in the `deployment` directory.
