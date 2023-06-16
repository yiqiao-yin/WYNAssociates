# Step Function Integrations

AWS Step Function Integrations
AWS Step Functions integrates with some AWS services so that you can call API actions, and coordinate executions directly from the Amazon States Language in Step Functions. You can directly call and pass parameters to the APIs of those services.

You coordinate these services directly from a Task state in the Amazon States Language. For example, using Step Functions, you can call other services to: 

- Invoke an AWS Lambda function

- Run an AWS Batch job and then perform different actions based on the results

- Insert or get an item from Amazon DynamoDB

- Run an Amazon Elastic Container Service (Amazon ECS) task and wait for it to complete

- Publish to a topic in Amazon Simple Notification Service (Amazon SNS)

- Send a message in Amazon Simple Queue Service (Amazon SQS).

- Manage a job for AWS Glue or Amazon SageMaker.

- Build workflows for executing Amazon EMR jobs

- Launch an AWS Step Functions workflow execution

## Service Integration Patterns

AWS Step Functions integrates with services directly in the Amazon States Language. You can control these AWS services using three service integration patterns:

Call a service and let Step Functions progress to the next state immediately after it gets an HTTP response. Read more about this pattern here: https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html#connect-default

Call a service and have Step Functions wait for a job to complete. Read more about this pattern here: https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html#connect-sync

Call a service with a task token and have Step Functions wait until that token is returned with a payload. Read more about this pattern here: https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html#connect-wait-token   