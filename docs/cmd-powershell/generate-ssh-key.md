# Generate SSH Key on Windows with Windows CMD/PowerShell

To get the most out of the GridPane platform, you’ll often find the need to use SSH to log into your server and use our GPCLI (GridPane Command Line Interface) commands. GPCLI a powerful set of tools that allow you to customize not only your server but your WordPress installations as well.

For security reasons, SSH access is only available with the use of an SSH key and is restricted to the root user.

## Step 1: Check if ssh client is installed

Make sure you have the latest updates of Windows if that is not possible, then at least you should have the  Windows 10 Fall 2018 build update. From this update, Windows 10 now comes with a built-in ssh client! To check if the client is working, fire up a Powershell or CMD window and type in this

```cmd
ssh
```

## Step 2: Create Your SSH Key Pair

Type the following command at the prompt then press enter.

```cmd
ssh-keygen -b 4096
```

When prompted for the file in which to save the key, press enter. The default location will be created.

Keep default values and no need for a pass phrase.

Congratulations! You now have an SSH key. The whole process will look like this:

What does all this mean?

The key generating process has created two files.

id_rsa (this is your private key, do not lose or give this to anybody!)

id_rsa.pub (this is your public key, you copy this to servers or give to others to place onto servers for you to authenticate against using your private key)

These keys are store by default in:

```cmd
C:\Users\WINUSER/.ssh/id_rsa.pub
```

## Step 3: Copy Your Public Key To Your Clipboard

We will use our good old notepad to get the contents of our public SSH key

You will need to run the following command. Remember to replace WINUSER with your own user

```cmd
notepad C:\Users\WINUSER/.ssh/id_rsa.pub
```
