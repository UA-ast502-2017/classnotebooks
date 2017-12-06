.. _coverage-label:

Code Coverage
=============
Here we will go over code coverage, the analysis of what lines of code are tested in our tests.

Our code coverage is set up using two different tools - `Coverage <https://coverage.readthedocs.io/en/coverage-4.3.4/>`__
and `Coveralls <https://coveralls.io/>`__. Coverage is what we use to report the coverage statistics on our code and
tests, while Coveralls is the service we use to hook our reports to our pipeline, giving us a website to read coverage reports for each build.

Coverage
--------
The documentation for the coverage package can be found `here <https://coverage.readthedocs.io/en/coverage-4.3.4/index.html>`__.

There are several different ways of reporting code coverage. I highly recommend reading the `How Coverage.py Works
<https://coverage.readthedocs.io/en/coverage-4.3.4/howitworks.html>`__ section to learn what it means to say your tests
have x% code coverage.
Basically, there are three phases to the code coverage we use:

* **Execution**: Executes code and records information.
* **Analysis**: Analyzes codebase for total executable lines.
* **Reporting**: Combines execution and analysis phases to give coverage report.

When tests are run, coverage.py runs a trace function that records each file and line that is executed when a test is
run. It records this information in a JSON file (usually) named ``.coverage``. This is called the execution phase.

During "Analysis," coverage looks at the compiled python files to get the set of executable lines, and filters through to
leave out lines that shouldn't be considered (e.g. blank lines, docstrings).

Finally, the Reporting phase handles the format in which to report its findings. There are several different outputs for the reports that you can use.

Configuration
^^^^^^^^^^^^^
Coverage also has a configuration file that allows the user to specify different options for coverage to handle, such as
multi-threading. The coverage configuration file is named ``.coveragerc`` by default. Information on the syntax for the
file can be found `here <https://coverage.readthedocs.io/en/coverage-4.3.4/config.html>`__. Through the configuration
file you can specify lines to skip, ignoring specific errors, where to output the coverage report, etc.

.. note::
    When running multiple coverage reports or using the multi-thread option, the command ``coverage combine`` is useful
    in that it will combine all the reports into one. Multi-threading will spawn multiple processes which will each
    have their own report so combining is very important for getting an accurate report. Note that all the reports must
    be in the same directory when running the command.


As a final note, it is important to note that, although code coverage is a great tool to have and use, it is not by itself
enough to say the code is bug free. 100% code coverage, in the end, does not mean much. It simply means all the executable
lines of code have been run in one way or another, but there is no real way to test `ALL` possible branches and situations
your code can take, especially for larger code bases. Read `this article <https://nedbatchelder.com/blog/200710/flaws_in_coverage_measurement.html>`__
for more on why code coverage can be flawed as well as a few examples. Just know that code coverage is a useful tool but
not fool-proof.


Coveralls
---------
Coveralls is the web service used to track our code coverage and report on our automated pipeline builds. Every time
code is pushed to our Bitbucket repo and our tests are run, we first obtain our report using coverage, then we send the
report to coveralls which in turn organizes our report with each build and displays the information for us on both the
coverage website and a badge on the bitbucket repo.

For information on how to setup a coveralls hook to a repo, look `here <https://github.com/coveralls-clients/coveralls-python>`__.
For our pipeline, we use Bitbucket Pipelines, so use the "Usage (another CI)" section.