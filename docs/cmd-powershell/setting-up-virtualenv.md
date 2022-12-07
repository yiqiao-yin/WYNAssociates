# Steps To Set Up Virtual Environment For Python On Windows

This blog is sourced from [here](https://www.c-sharpcorner.com/article/steps-to-set-up-a-virtual-environment-for-python-development/).

Notice how "python --version" returns the version to be "Python 3.7.0" and not "Python 3.6.6". The reason for this is the value of the PATH environment variable. To see the value you need to,
- right-click Computer and select Properties
- select Advance System Settings
- from the pop-up box select Advanced tab
- select Environment Variables 
- select PATH and then Edit

You can now copy the value in your favorite text editor and look at it. Mine looks like the following.

```cmd
C:\Python\Python37\Scripts\;C:\Python\Python37\;C:\Python\Python36\Scripts\;C:\Python\Python36\;    
//and a few other values
```

## Setting Virtual Environment

To set up a virtual environment, we first need to install the package virtualenv using pip. To do so, open up your PowerShell and execute the following commands.

```cmd
// upgrade pip to its latest version  
python -m pip install --upgrade pip   
  
// install virtualenv  
pip install virtualenv  
```

If your requirement falls under any of the following categories,
- have only one Python version installed
- don't want to specify any Python version
- want to use default Python version (check your version by running "python --version" on the command line)

Then, you can simply create your virtual environment using the "virtualenv venv" command, where "venv" is the environment name. However, if none of the above categories satisfies your requirement, then follow along as it's time to create your virtual environment using with Python 3.6. 

```cmd
// navigate to Desktop  
cd .\Desktop  
  
// create a new directory 'project-36'  
mkdir project-36  
  
// change currenct directory to 'project-36'  
cd .\project-36  
  
// create a virtual environment named 'venv', feel free to name it anything you like
virtualenv venv -p C:\Python\Python36\python.exe  
```

Notice the last command. With the attribute "-p" we have specified the Python version that we want our virtual environment to use. In our case, it's Python 3.6, which we had installed at "C:\Python\Python36". If you installed it at a different location, please pass the complete path here.
 
If you want to use Python 3.7 instead, all you need is to change the installation path for Python in the last command. Just put in the path where you installed Python 3.7 and you are good.
 
Now, it's time to activate the environment, check the Python version and also list the default packages installed for us. To do so, execute the following commands and you should see a similar output as shown in the image that follows. The (venv) on the left shows that our virtual environment is active. 

```cmd
// activate the virtual environment, see comments
.\venv\Scripts\activate
 
// check the python version
python --version
 
// list all packages installed by default
pip list
 
// deactivate the virtual environment
deactivate
```

## Different Syntax

*Comment*: You’ll need to use different syntax for activating the virtual environment depending on which operating system and command shell you’re using.

On Unix or MacOS, using the bash shell: source /path/to/venv/bin/activate
On Unix or MacOS, using the csh shell: source /path/to/venv/bin/activate.csh
On Unix or MacOS, using the fish shell: source /path/to/venv/bin/activate.fish
On Windows using the Command Prompt: path\to\venv\Scripts\activate.bat
On Windows using PowerShell: path\to\venv\Scripts\Activate.ps1

## Execution Policy Error

```cmd
\\ run the following to change the execution policy
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
```

Then run the command line to activate virtual environment.

Congratulations! You have successfully created your first virtual environment for Python. And, you are now all set to start your journey with Python development over Windows. 
