.. _using-label:

Working With Docker
===================

Here you will learn how some basics on working with Docker. 

Using Local Files
^^^^^^^^^^^^^^^^^
Once you have your image, you can `cp` over local files into the container. To do this you have to use the ``attach`` command and ``-d`` flag like so ::

        $ docker run -it -d simonko/pyklip:latest 

        exit

        $ docker cp <source file/directory> <container name>:<destination>

        $ docker start <container name>

        $ docker attach <container name>

It should be noted that if the specified destination does not exist, it will create the destination for you. For example if I were to do the following ::
        
        $ docker cp <somefile/directory> zealous_goldwasser:/pyklip

inside the `zealous_goldwasser` container and it did not already have a pyklip directory, docker would create the directory for me and place the file in it, just like the normal `cp` command.


Deleting Images and Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You may find that your docker is getting a bit cluttered after playing around with it. The following section will show you how to delete images and containers. You can also refer to `this cheat sheet <https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes#a-docker-cheat-sheet>`__ for more on deleting images and containers. The below is just a few basic and useful commands. 

`Deleting Containers`
"""""""""""""""""""""

To delete a container, first locate the container(s) you wish to delete, then use ``docker rm <ID or NAME>`` to delete::

        $ docker ps -a

        CONTAINER ID        IMAGE                   COMMAND                  CREATED             STATUS                     PORTS               NAMES
        c6695e4d9a63        simonko/pyklip:latest   "/usr/bin/tini -- ..."   6 seconds ago       Exited (0) 3 seconds ago                       zealous_goldwasser

        $ docker rm <container ID (c6695e4d9a63) or Name (zealous goldwasser)>

To delete multiple containers at once use the filter flag. For example, if you want to delete all exited containers ::

        $ docker rm $(docker ps -a -f status=exited -q)

You can also find all containers all exited containers using just the command in the parenthesis without the `-q` flag. This is particularly useful if there are many exited containers and you don't remember which ones you wanted to delete.

`Deleting Images`
"""""""""""""""""

To delete your images first you must find which ones you wish to delete. It should also be noted that to delete an image, there can be no containers associated with it. You must delete all containers from the image before deleting the image. ::


        $ docker images

        REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
        pyklip-pipeline         latest              e9a584c685bb        13 days ago         2.37 GB
        simonko/pyklip          latest              e9a584c685bb        13 days ago         2.37 GB
        localrepo               latest              dc74a96e5ef0        2 weeks ago         2.25 GB
        ubuntu                  latest              0ef2e08ed3fa        3 weeks ago         130 MB
        continuumio/anaconda3   latest              26043756c44f        6 weeks ago         2.23 GB

        $ docker rmi <repository name>

.. note::
        Before you delete an image, all containers using the image must be DELETED, not exited.

To delete ALL of your images ::

        $ docker rmi $(docker images -a -q)
