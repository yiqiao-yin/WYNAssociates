# API Gateway to Lambda

One can trigger a test run to test a lambda function. This assumes a lambda function exists in the AWS platform. One needs to set up the correct **API Gateway** to be able to trigger the run.

As a summary, what I did to replicate test procedure is:

1. Go to **API Gateway** > **API Keys** > **Actions** > **Create API Keys** (here I followed a template `username-projectname-api-Sandbox`

2. So once this is done, I have `yiqiao-projectname-api-Sandbox` and my own ID and API Key

3. Then go to a `test_lambda_prediction.py` to change the **ID** and **API Key**. A fake script can be seen below that is cited from [this source](https://www.mrnice.dev/posts/testing-aws-lambda-and-api-gateway/).

```py
def test_apigateway(dta: Path):
    payload = {"data": "data"}
    headers = {
        "Content-Type": "application/json", # or whatever used in creating API Gateway
        "Accept": "application/json", # or whatever used in creating API Gateway
        "x-api-key": "<ENTER_YOUR_API_KEY_HERE>",
    }
    response = requests.post(
        url="<PLEASE ENTER THE INVOKE URL FROM API GATEWAY AND MAKE SURE CORRECT DIRECTORY TO CALL LAMBDA FUNCTION IS PROVIDED>",
        headers=headers,
        json=payload,
    )
    print(response.json())


if __name__ == "__main__":
    dta_path = Path(
        r"<SOME SAMPLE PATH TO DATA>"
    )
    test_apigateway(dta_path)
```

4. Next, go to terminal (powershell cuz I have windows), to create a virtual environment

```py
virtualenv .venv
```

5. Install all required libraries

```py
pip list
pip install abc def
```

6. cd xxx/xxx inside to the tests folder, then run the folder

```py
cd xxx/xxx
py -m 
```

7. Then check `AWS CloudWatch` to make sure output from the console can be seen in the `CloudWatch`