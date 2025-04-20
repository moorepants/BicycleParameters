These are the instructions for updating the package.

1. Merge all commits from changes.
2. Update the CHANGELOG.
3. Update the dependency version info in setup.py, the README, and the docs
   description.rst.
4. Update the version number in bicycleparameters/version.py
5. Make sure the docs build.
6. Build the sdist and check contents ``python setup.py sdist``.
7. Git tag for the new version with ``git tag -a "Version X.X.X``.
8. Build the sdist ``python setup.py sdist``.
9. ``twine updload dists/<sdist>.tar.gz``
