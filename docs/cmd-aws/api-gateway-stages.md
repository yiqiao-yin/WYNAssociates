# API Gateway Stages, deployments, invoking, Postman

## API Gateway Stages and Deployment

Once you create a REST API in API Gateway, it doesn’t automatically become available to invoke. You need to publish the API first. In API Gateway, you publish the API to a stage.

A stage is a named reference to a deployment, which is a snapshot of the API. You use a Stage to manage and optimize a particular deployment. For example, you can configure stage settings to enable caching, customize request throttling, configure logging, define stage variables, or attach a canary release for testing.

Every time you make a change to your API, you must deploy it to a stage for that change to go live. You can host multiple versions of your API simultaneously by deploying changes to different stages. 

Using stages is perfect for setting up dev, qa, and production environments for you API. You can deploy your API to the appropriate stage as it moves through the software development lifecycle.

Read more about stages at: https://docs.aws.amazon.com/apigateway/latest/developerguide/rest-api-publish.html

## Invoking your API 

Once an API is published to a stage, an invoke URL is given to the API for that specific stage. Each stage gets it’s own invoke URL. If you are interesting in using custom domains for the invoke url, click here: https://docs.aws.amazon.com/apigateway/latest/developerguide/how-to-custom-domains.html  

## Postman

Postman is a collaboration platform for API development. Postman includes features for an API Client, Automated testing, Designing and Mocking APIs, Documentation, and more. Postman is very popular in the software development space, and it’s useful specifically with API development and testing. We focused on the API Client in this course, to download and try out Postman for yourself click here: https://www.postman.com/downloads/

