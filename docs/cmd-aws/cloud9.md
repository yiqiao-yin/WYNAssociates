# Cloud9, AWS APIs, AWS CLI

## AWS APIs

Everything in AWS is an API call and every AWS Service has its own set of APIs that you interact with. When you send HTTP requests to AWS, you sign the requests so that AWS can identify who sent them. You sign requests with your AWS access key, which consists of an access key ID and secret access key. You need to sign HTTP requests only when you manually create them. When you use the AWS Command Line Interface (AWS CLI) or one of the AWS SDKs to send requests to AWS, these tools automatically sign the requests for you with the access key that you specify when you configure the tools. When you use these tools, you don't need to learn how to sign requests yourself. 

To read about how requests need to be signed click here: https://docs.aws.amazon.com/general/latest/gr/signing_aws_api_requests.html

## AWS Command Line Interface

The AWS Command Line Interface (AWS CLI) is an open source tool that enables you to interact with AWS services using commands in your command-line shell. 

Installation of the AWS Command Line Interface is fairly straightforward, and if you’re using Amazon EC2 instances or AWS Cloud9, the tools are already installed for you.

After configuration, the AWS CLI enables you to start running commands that implement functionality equivalent to that provided by the browser-based AWS Management Console from the command prompt in your favorite terminal program:

Linux shells – Use common shell programs such as bash, zsh, and tcsh to run commands in Linux or macOS.

Windows command line – On Windows, run commands at the Windows command prompt or in PowerShell.

Remotely – Run commands on Amazon Elastic Compute Cloud (Amazon EC2) instances through a remote terminal program such as PuTTY or SSH, or with AWS Systems Manager.

Read information about the AWS CLI at: https://aws.amazon.com/cli/ 

Read information about installing the AWS CLI at: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html  

Read information about configuring the AWS CLI at: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

## AWS Cloud9

AWS Cloud9 operates as a cloud-based IDE, and has a wide variety of functionality already built in to help you with the development of and collaboration of your projects. A particular area where Cloud9 can assist is when working with the AWS Command Line Interface. Because the CLI tools are already installed in your environment, all you need to do is configure them to get started. 

You access the AWS Cloud9 IDE through a web browser. You can configure the IDE to your preferences. You can switch color themes, bind shortcut keys, enable programming language-specific syntax coloring and code formatting, and more.  

Read more about Cloud9 at: https://docs.aws.amazon.com/cloud9/latest/user-guide/welcome.html

AWS Cloud9 managed temporary credentials have permission restrictions on their own which may restrict some API calls you are using. You can find the list of those restrictions at: 

https://docs.aws.amazon.com/cloud9/latest/user-guide/how-cloud9-with-iam.html#sec-auth-and-access-control-temporary-managed-credentials

## AWS Credentials and Programmatic Access

Read about best practices with AWS Credentials here: https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html