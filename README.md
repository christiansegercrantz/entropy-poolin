# entropy-pooling

This library contains the Python function library for running entropy pooling of market scenarios for portfolio optimization goals. Further documentation is found in the PDF report. The report contains information on the input data and discussion regarding the method. We have separate instructions for filling the views data, see *views_data_instructions.md*.

*main.ipynb* is the file that is used to run the entropy pooling procedure (this will call the functions). There are also some example scripts for visualization.

The *entropy_pooling* package contains
1) *views.py* for uploading the views data from xlsx-file into linear optimization constraints

2) *entropy_minimizer.py* for performing the entropy minimization which yields the posterior distribution for the scenario probabilities

3) *markowitz_optimizer.py* for performing the Markowitz portfolio optimization task to find optimal allocations