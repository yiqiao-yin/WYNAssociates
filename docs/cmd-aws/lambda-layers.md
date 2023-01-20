# Lambda Layers, Performance Tuning, Best Practices

## AWS Lambda Layers

You can configure your Lambda function to pull in additional code and content in the form of layers. A layer is a ZIP archive that contains libraries, a custom runtime, or other dependencies. With layers, you can use libraries in your function without needing to include them in your deployment package.

Layers let you keep your deployment package small, which makes development easier.

Read more about layers here: https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html

## AWS Lambda Performance and Pricing

Following best practices with your AWS Lambda functions can help provide a more streamline and cost-efficient utilization of this component within your workflows. Make sure to keep the Lambda pricing calculator bookmarked to help provide estimations about how changes in your function build and utilization might affect the cost of your service usage.

Find the pricing calculator here: https://s3.amazonaws.com/lambda-tools/pricing-calculator.html

## AWS Lambda Power Tuning

The efficiency and cost of your lambda function often times relies on the amount of CPU and memory you have given you function. The more power you give a function, the more it costs to run. That being said, it can often be cheaper to run a function with more power. The reason for this is that your code runs faster with more CPU and memory available, so it can be a good exercise to do Lambda Power Tuning to find the best settings for your lambda function. 

AWS Lambda Power Tuning is an AWS Step Functions state machine that helps you optimize your Lambda functions in a data-driven way. 

You can provide any Lambda function as input and the state machine will run it with multiple power configurations (from 128MB to 3GB), analyze execution logs and suggest you the best configuration to minimize cost or maximize performance.

The state machine will generate a dynamic visualization of average cost and speed for each power configuration.

Find the AWS Lambda Power Tuning project here: https://serverlessrepo.aws.amazon.com/applications/arn:aws:serverlessrepo:us-east-1:451282441545:applications~aws-lambda-power-tuning

## AWS Lambda Best Practices

To read a list of AWS Lambda Best Practices in detail click here: https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html

