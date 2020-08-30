.. _contributing:

Contributing
============


#. Fork it
#. Run tests with ``pytest tests/ -vvvv``
#. Make your changes
#. Run tests again
#. Submit a pull request

Tips:


* If you want to, you can create an issue first. Ask any questions by opening a new issue.
* If you're encountering/fixing a bug and you're stuck, try clearing the cache. For example, a bug might have downloaded to the wrong folder, but after fixing the bug, you need to clear the cache, otherwise it would not download anything and display the wrong contents.

Unit tests
----------

Run ``pytest tests/ -vvvv`` for pytests. Run doctests with ``python -m doctest -v matrix/matrix.py``


Build and upload to PyPI
------------------------

(Not done yet)

#. Run integration tests locally
#. Review github action logs to make sure nothing is wrong
#. Bump version info in ``setup.py``\
#. Run:

.. code-block:: sh

   python setup.py sdist bdist_wheel
   twine upload dist/*
