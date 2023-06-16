# Test Lambda Script

In some situation, one needs to test a lambda script locally. This section of markdown file assumes that there is an existing lambda function written and created. In addition, there is a test script locally called `test_lambda.py`. This tutorial walks readers to how to create **API Gateway** to invoke the lambda function. 

## Here are the steps (the goal is to run a test lambda py script from virtual environment and invoke lambda function):

1. Go to **API Gateway** > **API Keys** > **Actions** > Create **API Keys** (here it is recommended to follow some template existed before)

2. So once this is done, I have my own version, for example, I can call it `yiqiao-projectname-api-location` and my own ID and API Key

3. Then go to test script, `test_lambda.py` to change the ID and API Key

3. Next, go to terminal (powershell cuz I have windows), to create a virtual environment

4. Install all required libraries in the virtual environment

5. Go inside to the testfolder where the test python script lives, run `cd xxx/xxx/test` and examine output

6. Last check **AWS CloudWatch** to make sure output from the console can be seen in the CloudWatch

Once this is set up, one can get the lambda script ready and hit the `test_lambda.py` script. Related print statements are added so CloudWatch produce desired error messages. Hence, we can test run iteratively until there are no bugs and all outputs are ideal.
