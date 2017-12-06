.. _dockerSetup-label:

Setup
=============

Here you will learn how to install and setup your Docker.

Installation
^^^^^^^^^^^^
We will be using the community edition of Docker.

For Ubuntu Linux ::

        sudo apt-get update
        sudo apt-get install docker-ce

For all other OSes, installation instructions and requirements can be found `here <https://docs.docker.com/engine/installation/>`__.


Setup
^^^^^^^^^^^^
From a fresh install, there are a few steps to getting your container up and running. 

1. Download and run the pyKLIP image. You can do this by pulling and running, but simply running the pyKLIP image will
do both steps in one. Executing the run command will first check your local machine for the appropriate images and use
them if Docker finds them, or download them from Docker Hub if it fails. For now we'll start with the pull command::

        $ docker pull simonko/pyklip

2. From here, to check if the appropriate image has been set up use the :mod:`docker images` command, and you should get
something similar to the following ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        simonko/pyklip          latest              e9a584c685bb        4 hours ago         2.37 GB

3. Running the command below creates a container of the pyklip image and gives us an interactive shell to interact with
container. The ``-i -t`` flags allows for interactive mode and allocates a pseudo-tty for the container respectively.
This is usually combined into the flag ``-it``. If you don't specify a tag, it'll generate some random name for you.
(ex. sad_lovelace, agitated_saha, ecstatic_pare, etc) ::

        $ docker run -it simonko/pyklip:latest /bin/bash

4. When you're done with the container, simply type ``exit`` and your session will end. If you get the message that
states there is a process running, simply type ``exit`` again and it'll exit the session.
5. After you've made your container you should be able to see it with ::
        
        $ docker ps -a

        CONTAINER ID        IMAGE                   COMMAND                  CREATED             STATUS                     PORTS               NAMES
        c6695e4d9a63        simonko/pyklip:latest   "/usr/bin/tini -- ..."   6 seconds ago       Exited (0) 3 seconds ago                       zealous_goldwasser
6. To get into the container, you have to first start the container again, then use the attach command to get back into
the interactive shell. ::

        $ docker start <container name>
        $ docker attach <container name>

For a very basic tutorial on Docker and how to use it, check out the Docker docs and tutorials `here <https://docs.docker.com/engine/getstarted/step_three/#step-2-run-the-whalesay-image>`__. There are a lot of helpful tutorials and information there. 
