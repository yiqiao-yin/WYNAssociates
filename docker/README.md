# README

This folder contains information about docker container and the instructions required to set up docker container for the optimal environment. A few good articles ([MicroFocus](https://www.microfocus.com/documentation/enterprise-developer/ed40pu5/ETS-help/GUID-F5BDACC7-6F0E-4EBB-9F62-E0046D8CCF1B.html), [SimpliLearn](https://www.simplilearn.com/tutorials/docker-tutorial/what-is-docker-container), [InfoWorld](https://www.infoworld.com/article/3310941/why-you-should-use-docker-and-containers.html), [DZone](https://dzone.com/articles/top-10-benefits-of-using-docker)) point out the benefit of using docker container in regards to portability, scalability, and efficiency in production.

- In this folder, there is a "dockerfile" which lists the command to run in PowerShell and it contains libraries I used to create this package. For general purpose, I recommend users to put down a list of whatever libraries they desire to have when building the docker container.
- In this folder, there is another file called "openslide" which is the base installation file I used for the "openslide-python" library. From my experience, it is easier to install the base file in Windows, and then use docker container to install the python adaptation. Other ways are possible, yet I have run into many bugs with other methods. For more information, please refer to [here](https://openslide.org/api/python/).

## Installation Manual

This is to test if a library you desire to have can run on a virtual environment using docker. Most of the libraries should work fine so this step can be skipped. In some scenarios where a library is originally not written in python, it is recommended to use the virtual Linux enrivonment to test it out before install a full docker container.
- Install the docker desktop [here](https://docs.docker.com/desktop/windows/install/) | After downloading "Docker Desktop for Windows", you will have a ".exe" file locally in your computer. Please install it.
- Open the docker desktop
- Open powershell
- RUN: docker run --rm -it --entrypoint bash openslide 
- Install whatever package you like using "pip install XXX" or "conda install XXX"
- Test it by "python" to open python from the powershell and then import that library
- If test is successful, then run the following to build the docker image

If the above procedure runs successfully, please follow the steps below to install the entire docker container. It is recommended to run the following in a Powershell.

Run
```
docker build -t openslide .
```

then run
```
docker run --name openslide -v WHATEVER_PATH_YOU_DESIRE:/mnt -d -p 8888:8888 openslide
```

for example, I did mine in my OneDrive folder inside C-disk (for organization usage of large-scale data files, this is more ideal because data can be saved on a server that can be accessed using docker container directly)
```
docker run --name openslide -v C:\Users\eagle\OneDrive:/mnt -d -p 8888:8888 openslide
```

This will permantly create a container in docker desktop that you can then launch when you open it, and the notebook will be located at localhost:8888. Please make sure to shutdown other notebook servers that might want that port before or you can change the first port in the –p command to be something different. In case a bug occurs that says no such directory is found or a 404 error is reported, it is recommended to check the port first.

After the above steps, you have succesfully installed the docker container. To open it, it is recommended to use the following steps.
1. Open Docker Desktop as Admin <= enter admin username and pw
2. In containers / apps, there should be a container called openslide => click on the openslide text
3. A log will show up and usually it will start. However, if it's not started, go to top right corner to click "start". Then there are more logs and in the bottom there should be a link that says "https://127.0.0.1:8888/lab?token=SOME_TOKEN_HERE"
4. Copy the last http link and paste in a browser 
5. Open up a jupyter notebook and you are done. 

After you finish your task and desire to close the container, it is recommended to use the following steps.
1. Make sure there is no running code
2. "Flie" => "Shut down"
3. Close the tab
4. Close the Docker container

In some scenario due to the setup of your computer, it might be more optimal to redo the steps again. This means instead of fixing the broken contianer it might be easier to just reinstall it again. In this case, it is recommended to use the following steps.
1. Delete image in the container
2. Uninstall Docker Container
3. Reinstall Docker Container
4. Start Docker
5. Go to powershell (as admin) to build

## Other Language

This [git](https://github.com/yeasy/docker_practice) has a docker documentation that is provided in Chinese. The online book can be accessed [here](https://vuepress.mirror.docker-practice.com/).
