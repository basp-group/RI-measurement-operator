Setup
=====

Installation
------------

Clone the current repository as follows.

.. code-block:: bash

   git clone https://github.com/basp-group/RI-measurement-operator.git
   cd RI-measurement-operator


Representative examples and tests
---------------------------------

To run a representative example, run the :mat:scpt:`example.m` script. An example use of the NUFFT interface considered in this library is provided in the last section of the script. 

To launch the tests, run the :mat:scpt:`runTest.m` file.


Contributing
------------

Source Code: `https://github.com/basp-group/RI-measurement-operator <https://github.com/basp-group/RI-measurement-operator>`_

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Make sure any new functionality is properly documented using the ``numpy`` docstring style.
- To build the documentation, issue the folowing commands.

.. code-block:: bash

   # setup conda environment to build the documentation
   conda env create --name mo-doc --file environment.yml 
   
   ## alternative using conda
   # conda create -n mo-doc
   # conda activate mo-doc
   # conda install pip
   # pip install miss_hit
   # pip install -r requirement.txt

   # building the documentation in html format
   cd docs
   make html

- All the generated ``.html`` files are contained in the ``docs/build`` folder.
- If needed, you can delete the ``conda`` environment as follows

.. code-block:: bash
   
   conda env remove -n mo-doc


Pushing the documentation online
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a ``worktree`` from the ``master`` branch

.. code-block:: bash

   # make sure the folder html does not exist before running the command
   git worktree add docs/build/html gh-pages
   cd docs/build/html
   git add .
   git commit -m "Build documentation as of $(git log '--format=format:%H' master -1)"
   git push origin gh-pages
   # delete the worktree
   cd ../
   git worktree remove html


Code layout
^^^^^^^^^^^

If you contribute code to the library (through a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_), make sure any submitted code is properly formatted with the `miss_hit <https://pypi.org/project/miss-hit/>`_ package using the provided ``miss_hit.cfg`` configuration file

.. code-block:: bash

   # activate mo-doc environment (see previous paragraph)
   conda activate mo-doc
   # run the following command from the root of the package (where the miss_hit.cfg file is)
   mh_style --fix .
