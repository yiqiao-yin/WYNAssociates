# Exercise 1: Setting Up and Exploring the SDK

In this exercise, you install and configure the AWS Command Line Interface (AWS CLI) and the AWS SDK for Python (Boto3). After you install the necessary requirements, you create an S3 bucket and deploy a web application to the bucket. You then set up data in Amazon S3, and configure AWS Systems Manager parameters for the Dragons application. After you create all the resources, you explore the Python application.

If you choose to use a local IDE instead of an AWS Cloud9 instance, you can follow the optional steps on Local IDE Prerequisites.

You will want to run the steps in the US East (N. Virginia) us-east-1 Region.

## Task 1: Creating an AWS Cloud9 environment

For this task, you will create an AWS Cloud9 environment, which you will use as your development environment throughout the exercises.

In the AWS Management Console, choose Services, and then search for and open Cloud9.

Choose Create environment.

For Name, enter Python-DevEnv and choose Next step.

In the Configure settings page, keep the default settings and choose Next step.

Choose Create environment. It might take a few minutes for the environment to be created.

In the AWS Cloud9 (or local IDE terminal), install the SDK for Python:

```cmd
pip install boto3
```

## Task 2: Creating an S3 bucket

In this task, you will create an S3 bucket. This bucket will store the web application frontend. AWS Command Line Interface (AWS CLI) commands are supplied for use in your AWS Cloud9 or local IDE terminal. To perform the same steps with the SDK, see the appendix.

1. In the AWS Cloud9 or local IDE terminal, run the following commands. You will be prompted for a bucket name, then the following command creates the bucket. You need to use a unique bucket name. For example, you could combine your initials and -dragons-app to create a bucket name (such as hjs-dragons-app).

First, save your bucket name to an environment variable:

```cmd
echo "Please enter a bucket name: "; read bucket;  export MYBUCKET=$bucket
```

Create the bucket:

```cmd
aws s3 mb s3://$MYBUCKET
```

If your bucket name isn’t unique, you will see an error message. Continue running the command and entering a new bucket name until you have successfully created a bucket.

2. After the bucket is created, make the environment variable available in future exercises by adding an entry to your `.bashrc`:

```cmd
echo "export MYBUCKET=$MYBUCKET" >> ~/.bashrc
```

3. List the buckets in your account and confirm that the newly created bucket is in the response.

```cmd
aws s3 ls
```

For more information about the SDK syntax, see [List existing buckets](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-creating-buckets.html) in the Boto3 Docs documentation.

For more information about AWS CLI commands, see the [AWS CLI Command Reference](https://docs.aws.amazon.com/cli/latest/index.html).


## Task 3: Deploying a web application to the S3 bucket

In this task, you will deploy a simple web application to the S3 bucket. Amazon S3 will become the web server for your static website, which includes HTML, images, and client-side scripts. AWS CLI instructions for copying the web application are supplied.

1. In the AWS Cloud9 or local IDE terminal, download and extract the web application.

```cmd
wget https://aws-tc-largeobjects.s3.amazonaws.com/DEV-AWS-MO-BuildingRedux/downloads/webapp1.zip
unzip webapp1.zip -d ~/webapp1
```

2. Copy all the contents of the `webapp1` folder to Amazon S3.

```cmd
aws s3 cp ~/webapp1 s3://$MYBUCKET/dragonsapp/ --recursive --acl public-read
```

3. To find the URL for the web application, run the following AWS CLI command.

```cmd
echo "URL: https://$MYBUCKET.s3.amazonaws.com/dragonsapp/index.html"
```

4. Visit the web application (which is hosted on Amazon S3) by copying the URL output into a browser window.

5. Confirm that your web application works. It will look similar to the following screen capture.

![image](https://aws-tc-largeobjects.s3.amazonaws.com/DEV-AWS-MO-BuildingRedux/images/Exercise-1-web-application.png)(Dragons application user interface)

6. In your browser, bookmark this application because you will return to it in future exercises.

## Task 4: Setting up the Dragons application

In this task, you will create resources that the Dragons application will use. You will download the file that contains the dragons data in JavaScript Object Notation (JSON) format, and upload this data to your S3 bucket. Then, you will create two parameters for Systems Manager. The console application that you create after this section uses the dragons data for querying, and it uses the parameters for configuration.

1. Download the source of the dragons data: `dragon_stats_one.txt`.

```cmd
wget https://aws-tc-largeobjects.s3.amazonaws.com/DEV-AWS-MO-BuildingRedux/downloads/dragon_stats_one.txt
```

2. Copy the dragons data in `dragon_stats_one.txt` to the root of the S3 bucket.

```cmd
aws s3 cp dragon_stats_one.txt s3://$MYBUCKET
```

3. Confirm that the data populated the S3 bucket. You should see the `dragonsapp/` prefix and a `dragon_stats_one.txt` object, in addition to the other objects in the bucket.

```cmd
aws s3 ls s3://$MYBUCKET
```

4. Set the Parameter Store value for `dragon_data_bucket_name`.

```cmd
aws ssm put-parameter \
--name "dragon_data_bucket_name" \
--type "String" \
--overwrite \
--value $MYBUCKET
```

5. Set the Parameter Store value for the `dragon_data_file_name`.

```cmd
aws ssm put-parameter \
--name "dragon_data_file_name" \
--type "String" \
--overwrite \
--value dragon_stats_one.txt
```

## Task 5: Exploring the console application

In this task, you will explore the console application.

1. Download and extract the console application, which is a Python file.

```cmd
wget https://aws-tc-largeobjects.s3.amazonaws.com/DEV-AWS-MO-BuildingRedux/downloads/app.py
```

2. Open the `app.py` file to see how it uses the Systems Manager parameters to retrieve the bucket name and file name.

3. You can run the file to get some dragon data from `dragon_data_file_name`.

```cmd
python3 app.py
```

It should return some JSON output that’s similar to the following:

```cmd
{"description_str":"From the northern fire tribe, Atlas was born from the ashes of his fallen father in combat. He is fearless and does not fear battle.","dragon_name_str":"Atlas","family_str":"red","location_city_str":"anchorage","location_country_str":"usa","location_neighborhood_str":"w fireweed ln","location_state_str":"alaska"}
```

