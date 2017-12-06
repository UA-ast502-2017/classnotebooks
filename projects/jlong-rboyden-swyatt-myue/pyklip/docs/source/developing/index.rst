.. _developing-label:

Developing for pyKLIP
=====================

Adding Modules
~~~~~~~~~~~~~~
pyKLIP is fairly modular and allow you to add modules for various functionality like support for different instruments or different
forward modelling methods. Here's some guides on how to make your own.

.. toctree::
   :maxdepth: 1

   add_instrument


Docker
~~~~~~
One very useful tool to have is a local build environment of the pyKLIP package for testing and validation purposes.
We will be using a software container platform called Docker and this tutorial will provide a brief overview on what it
is, how to set it up, and how to use it for pyKLIP.

Here you will find everything you need to know about Docker for pyKLIP.

.. toctree::
   :maxdepth: 2

   setup
   using
   sharing

Tests
~~~~~
Here we will lay out the testing infrastructure used for pyKLIP.

.. toctree::
   :maxdepth: 2

   tests
   coverage