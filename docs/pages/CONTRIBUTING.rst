.. _contributing:

Contributing
============


#. Fork it
#. Run tests with ``pytest testing/ -vvvv -l``
#. Make your changes
#. Run tests again (add ``-s --inte`` for integration tests if you want)
#. Submit a pull request

Tips: 


* If your git client complains about committing to master, just remove ``.pre-commit-config.yaml``
* If you want to, you can create an issue first. Ask any questions by opening a new issue.
* If you're encountering/fixing a bug and you're stuck, try clearing the cache. For example, a bug might have downloaded to the wrong folder, but after fixing the bug, you need to clear the cache, otherwise it would not download anything and display the wrong contents.

Unit tests
----------

Run ``pytest tests/ -vvvv -l``. Add ``-s --inte`` for integration testing, but don't be surprised if it fails, because integration tests require a valid config/account + internet connection

Run doctests with ``python -m doctest -v {file}``

Build and upload to PyPI
------------------------


#. Run integration tests locally
#. Review github action logs to make sure nothing is wrong
#. Bump version info in ``__init__.py``\ , ``setup.py``\ , and ``CONTRIBUTING.md``
#. Run:

.. code-block:: sh

   # Change 1st argument to where [`plantuml.jar`](https://plantuml.com/download) is stored
   java -jar ~/Applications/plantuml.jar docs/puml/classes -o render
   python setup.py sdist bdist_wheel
   twine upload dist/*
   pip install koneko --upgrade
