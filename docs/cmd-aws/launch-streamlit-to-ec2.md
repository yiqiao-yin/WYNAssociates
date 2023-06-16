## EC2 Instance Setup for Streamlit Application

This guide outlines the steps to launch a Streamlit application using an EC2 instance on AWS. By following these steps, you can deploy your application and make it accessible via a URL.

### Launching an EC2 Instance

1. Go to the AWS website and navigate to EC2.
2. Launch an instance, selecting the default Linux system configuration.
3. Make sure to select private subgroup network and select `SSH_COMPANYNAME` and `ALL_COMPANYNAME` as options; there should be no public network access. For public ports and URL requests, one can refer to this [video](https://youtu.be/904cW9lJ7LQ) for detailed explanation.
4. Download the `.pem` file provided during the instance creation process. This is the key needed to connect to the instance later.

### Connecting to the EC2 Instance

1. After the instance is launched, navigate to the "Connect" page.
2. Use the code provided to connect to the Linux system in the instance from your local laptop terminal or Git Bash using the command `ssh -i "name_of_key_file.pem" ec2-user@this_is_your_private_ip_address`.
3. You should see a symbol indicating a successful connection.

### Copying Files to the EC2 Instance

1. Ensure that the following files exist in a local directory: `packagename1.tar.gz`, `packagename2.tar.gz`, `app.py` (the Streamlit application), `requirements.txt`, and the `.pem` file.
2. In the terminal, create a virtual environment using `python3 -m venv myenv` and activate it using `source myenv/bin/activate`. Sometimes a new Linux system may not have `pip` installed, so you'll need to run this code `sudo apt-get install python3-pip` or `sudo yum install pip` in order to get `pip` for a new Linux system. 
3. Copy the files to the Linux system using the command `scp -i "name_of_key_file.pem" app.py ec2-user@this_is_your_private_ip_address:/home/ec2-user/`, substituting `app.py` with the names of the other files.
4. Install the dependencies using the command `pip install -r requirements.txt`.

### Running the Streamlit Application

1. Run the application using `streamlit run app.py` in the terminal.
2. To run the application in the background, use `streamlit run app.py &`.
3. To shut down the application, connect to the instance via SSH and use the command `ps -ef | grep streamlit` to locate the app and shut it down by running `NUMBER_TO_YOUR_RUNNING_INSTANCE kill` command.