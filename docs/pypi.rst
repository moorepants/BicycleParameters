These are the instructions for updating the package.

1. Merge all commits from changes.
2. Update the version info in the README
3. Update the version number in bicycleparameters.__init__.py
4. Update the version number in docs/conf.py.
5. Make sure the docs build.
6. Git tag for the new version.
7. python setup.py regsiter sdist upload
8. Upload the Documentation files.
9. Upload the basic data files if you've edited them.
