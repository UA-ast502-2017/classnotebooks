.. _tests-label:


#######
Testing
#######

Here we will go over how we test and what our testing infrastructure looks like for pyKLIP.

All of our tests can be found in the ``tests`` folder located in the base directory. In the folder, each module or feature gets it's own test
file, and inside every file, each function is a different test for the module/feature.

The testing workflow for pyKLIP can be broken down into the following steps:

* Creating the tests
* Documenting the tests
* Running the tests


Creating Tests
==============
All tests for pyKLIP can be found in the ``tests`` directory. We use pytest to run all of our tests in this directory.
All tests should be named "test_<module/purpose>", and within the test files, each function should be named "test_<function
name>" to give an idea of what the test is for. The docstring for the function will go into detail as to what the test
is testing and a summary of how it works.

Our testing framework is organized so that each file tests an individual module or feature, and each function inside
each test file tests different aspects of the module/feature.

During the test, you may find it necessary to look at input or output files. In this case, all pathing should be agnostic of the directory structure outside of the pyKLIP folder.
It is suggested you first construct relative paths with respect to the current test file or the pyklip base directory, and then convert it to an absolute path before
using it in the function.

Some commands you may find helpful to find files:

* **os.path.abspath(path)** - returns the absolute path of the path provided
* **os.path.dirname(path)** - returns the name of the directory of the filepath provided.
* **os.path.exists(path)** - returns True if the path exists, False otherwise.
* **os.path.sep** = path separator. This is important because different OSes can have different path separators. For example Ubuntu Linux uses "/" while Windows uses "\\". This will take care of that.
* **os.path.join(args)** - returns a string with all the args separated by the appropriate path separator. For exmaple ``os.path.join("this", "is", "a", "path")`` would return ``"this/is/a/path"`` in Ubuntu Linux.
* **__file__** - When used on its own, filepath of this python file. All python modules should also have this as an attirbute (e.g. ``pyklip.__file__``)


Documenting Tests
=================
Docstring for tests should follow the Google Python stylguide. Here is an exmaple of a function docstring::

    """
    Summary of what your tests does goes here.

    Args:
        param1: First param.
        param2: Second param.
        etc: etc

    Returns:
        A description of what is returned.

    Raises:
        Error: Exception.
    """

Use the `following link <http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`__ for more details on
docstrings as well as Python style in general.

Running Tests
=============
All of our tests are run automatically using ``pytest`` on a Docker image using a continuous integration build system (Bitbucket Pipelines). 
This allows us to test pyKIP against a fresh and updated Python installation to ensure functionality is not broken and is comptable with the newest Python version.
If these terms seem unfamiliar, please refer to our :ref:`developing-label` page under the "Docker" section for more
information on Docker.

Here is a simple overview of the steps invovled in our automated testing framework:

1. Bitbucket Pipelines reads our pipeline yml file to build the pipeline.
2. Creates a docker image of the latest continuum anaconda3.
3. Git clones the pyklip repository inside image.
4. Installs all necessary packages.
5. Runs tests using pytest on the test directory.
6. Runs coverage analysis on our tests.
7. Submits coverage report.

You can also run tests locally. This is typically useful when you make changes and want to check that the changes does not break any functionality.
It can also be useful if you write a test before writing the function code, and debug your code as you develop your function. That way, you will
have validation code from the start. In this case, you may not want to run the full suite of tests. 

To simply run a single test you can either call the file directly using::

    $ python <path/to/test file name>.py

To run all tests simply call::

    $ pytest

The general command for pytest is as follows and there are two ways to invoke it::

    $ python -m pytest [args]
    $ pytest [args]

The above line will invoke pytest through the Python interpreter and add the current directory to sys.path. Otherwise
the second line is equivalent to the first.
There are many arguments and many different ways to use pytest. To run a single test simply enter the path to the test
file to run, to test all files in a directory use the path to the directory instead of a single file.
For more information on how to use pytest and some of its various usages, visit `this link <https://docs.pytest.org/en/latest/usage.html#>`__.
