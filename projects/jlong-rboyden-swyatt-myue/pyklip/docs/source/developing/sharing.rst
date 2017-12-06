.. _sharing-label:

Sharing Images
===================

Here you will learn how to create and upload your own images onto Docker Hub for others to use. 

Creating Images
^^^^^^^^^^^^^^^
In this section, you will learn how to create and upload your own image. To do this you need to make a dockerfile. If you wish to share the image for others to use, you need to create a Docker Hub account and push your image into a repository. This section will go over all of these steps. For a more detailed tutorial `use this link <https://docs.docker.com/engine/getstarted/step_four/#step-4-run-your-new-docker-whale>`__. Otherwise here are the very basics. 

Docker images are created from a set of commands in a dockerfile. What goes on this file is entirely up to you. Docker uses these commands to create an image, and it can be an entirely new one or an image based off of another existing image. 


1. Create a file and name it dockerfile. There are three basic commands that go on a dockerfile.
    - FROM <Repository>:<Build> - This command will tell docker that this image is based off of another image. You can specify which build to use. To use the most up-to-date version of the image, use "latest" for build. 
    - RUN <Command> - This will run commands in a new layer and creates a new image. Typically used for installing necessary packages. You can have multiple RUN statements.
    - CMD <Command> - This is the default command that will run once the image environment has been set up. You can only have ONE CMD statement.

    For more information on RUN vs CMD here is a `useful link <http://goinbigdata.com/docker-run-vs-cmd-vs-entrypoint/>`__.

2. After you've made your file run the following command to create your image ::
    
        $ docker build -t <Image Name> <Path to Directory of Dockerfile>

The ``-t`` flag lets you name the image. 

For example, the docker file used for the pyklip image I set up above (under the "Using Docker" section) is made using a dockerfile with the following content: ::

        FROM continuumio/anaconda3:latest
        RUN git clone https://bitbucket.org/pyKLIP/pyklip.git \
         && pip install coveralls \
         && pip install emcee \
         && pip install corner \
         && conda install -c https://conda.anaconda.org/astropy photutils

Uploading Images
^^^^^^^^^^^^^^^^
1. If you haven't already, `create a Docker Hub account <https://hub.docker.com/register/?utm_source=getting_started_guide&utm_medium=embedded_MacOSX&utm_campaign=create_docker_hub_account>`__. 
2. After you've made your account, sign in and click on "Create Repository" and fill out the details. Make sure visibility is set to PUBLIC. Press create.
3. Find your image ID. Using a previous example ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB

The image ID would be e9a584c685bb. 

4. Tag the image using ::
        
        $ docker tag <Image ID> <DockerHub Account Name>/<Image Name>:<Version or Tag>

So for the pyklip pipeline image my command would be: ::
        
        $ docker tag e9a584c685bb simonko/pyklip:latest 

Check that the image has been tagged ::

        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB
        simonko/pyklip          latest              e9a584c685bb        13 days ago         2.37 GB

5. Login to Docker on terminal ::
        
        $ docker login

        Username: *****
        Password: *****
        Login Succeeded
6. Push your tagged image to docker hub ::

        $ docker push <Repository Name> 

7. To pull from the repo now, all you have to do is run the repo. Docker will automatically pull from docker hub if it cannot find it locally. 
