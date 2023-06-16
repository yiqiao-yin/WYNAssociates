
## What is the purpose of using AWS?

AWS (Amazon Web Services) is a cloud computing platform that provides a wide range of services and solutions for various use cases. The purpose of using AWS is to:

Reduce IT infrastructure costs: AWS provides on-demand access to computing resources, reducing the need for organizations to invest in and maintain their own physical servers.

Increase scalability and flexibility: AWS allows organizations to quickly and easily scale their computing resources as needed, making it ideal for businesses that experience spikes in demand.

Enhance disaster recovery and business continuity: AWS provides robust disaster recovery and business continuity solutions that ensure organizations can quickly recover from any disruption.

Improve application performance: AWS offers a range of performance-enhancing services, such as caching, content delivery networks, and auto-scaling, that help improve the performance and reliability of applications.

Speed up innovation: AWS provides a wide range of services and solutions that enable organizations to quickly and easily innovate and bring new products and services to market.

Overall, the purpose of using AWS is to help organizations reduce costs, increase agility, and improve their ability to compete in the digital economy.

### Sample AWS Lambda Function

Here's an example of a "Hello World" AWS Lambda function written in Python:

```py
def lambda_handler(event, context):
    print("Hello World")
    return "Hello World"
```

The function will be saved in `app.py` script in the same directory of the `Dockerfile`.

In this example, the `lambda_handler` function takes two arguments: `event` and `context`. The event argument contains information about the triggering event, such as an API request. The `context` argument contains information about the current execution context, such as the runtime and the AWS request ID.

The function simply returns a dictionary with a single key-value pair, where the key is "message" and the value is the string "Hello World!".

This function can be uploaded to AWS Lambda and triggered by various events, such as an API request or a schedule, to execute the code and return the message "Hello World!".

## How can dockerfiles help us launch images to AWS

Dockerfiles can be used to build Docker images, which can then be used to launch instances on AWS. The steps to launch a Docker image on AWS are as follows:

Create a Dockerfile: Write the instructions to build the Docker image in a Dockerfile.

Build the Docker image: Run the "docker build" command to build the Docker image based on the instructions in the Dockerfile.

Push the Docker image to a registry: Push the newly built Docker image to a registry, such as Docker Hub or Amazon Elastic Container Registry (ECR), from where it can be pulled to launch instances on AWS.

Launch instances on AWS: Use the AWS Management Console, AWS CLI, or AWS SDKs to launch instances on AWS. Select the Docker image from the registry and specify the desired instance type and network configuration.

Manage the instances: Use the AWS Management Console, AWS CLI, or AWS SDKs to manage and monitor the instances. This includes tasks such as scaling the number of instances, monitoring resource utilization, and updating the Docker images.

By using Dockerfiles to build and launch Docker images on AWS, organizations can benefit from the scalability and reliability of the AWS infrastructure, while retaining the flexibility and portability of Docker containers.

## Sample Dockerfile

Here's an example of a Dockerfile that executes a Python script:

```docker
# python3.8 lambda base image
FROM public.ecr.aws/lambda/python:3.8

# copy requirements.txt to container
COPY requirements.txt ./

# installing dependencies
RUN pip3 install -r requirements.txt

# Copy function code to container
COPY app.py ./

# setting the CMD to your handler file_name.function_name
CMD [ "app.lambda_handler" ]
```

About this dockerfile, what does the directory look like the following. Notice that the `CMD` line (or the last line of the `Dockerfile` the "app" has a method that is the same name as the `lambda_handler` function in `app.py` script).

```cmd
/
|-- app.py
|-- Dockerfile
```

The `hello.py` file contains the "Hello World" Python script, which will be executed when the Docker container is launched. The `Dockerfile` specifies the instructions for building the Docker image.

When you run the docker build command in this directory, the Docker engine will read the Dockerfile and use its instructions to create the Docker image. Once the image is built, you can launch a container from it to run the "Hello World" Python script.

## Retrieve an authentication token and authenticate your Docker client to your registry

```py
# Retrieve an authentication token using the AWS CLI
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

# Authenticate the Docker client to the registry
docker login -u AWS -p <authentication-token> 123456789012.dkr.ecr.us-west-2.amazonaws.com
```
In this example, the "aws ecr get-login-password" command is used to retrieve an authentication token for the registry located at "123456789012.dkr.ecr.us-west-2.amazonaws.com". The authentication token is piped to the "docker login" command, which authenticates the Docker client to the registry.

The "-u" option is used to specify the username (AWS), and the "-p" option is used to specify the authentication token. The registry URL is specified at the end of the command.

Once the Docker client is authenticated, you can use the "docker push" and "docker pull" commands to push and pull images to and from the registry.

## Build Image in ECR Using Docker

Here we use Docker to build image and push the image to ECR. This section assumes you have `Docker Desktop` installed and opened locally. 

Go to AWS ECR and you can `View Commands` to see the following. Make sure that you have the latest version of the AWS Tools for PowerShell and Docker installed. For more information, see Getting Started with Amazon ECR. 

1. Retrieve an authentication token and authenticate your Docker client to your registry.
Use AWS Tools for PowerShell:

```cmd
(Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin XXX.dkr.ecr.us-east-1.amazonaws.com
```

2. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:

```cmd
docker build -t NAME_OF_YOUR_IMAGE .
```

3. After the build completes, tag your image so you can push the image to this repository:

```cmd
docker tag NAME_OF_YOUR_IMAGE:latest XXX.dkr.ecr.us-east-1.amazonaws.com/NAME_OF_YOUR_IMAGE:latest
```

4. Run the following command to push this image to your newly created AWS repository:

```cmd
docker push XXX.dkr.ecr.us-east-1.amazonaws.com/NAME_OF_YOUR_IMAGE:latest
```

## Create Lambda Function Using Image in ECR

Once you have imaged pushed to a container in ECR on AWS, you can go to `Lambda` to create a new function. You need to find the right `Image URL` and copy that to be able to set up the Image in `Lambda` function. 

After this is done, you can trigger a `Test` in `Lambda` function to see if the function works or if you need to debug. 