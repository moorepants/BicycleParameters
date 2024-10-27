These are the instructions for updating the package.

1. Merge all commits from changes.
2. Update the version info in the README
3. Update the version number in bicycleparameters/version.py
4. Update the version number in docs/conf.py.
5. Make sure the docs build.
6. Git tag for the new version.
7. python setup.py register sdist upload
8. Upload the Documentation files.

   - cd docs/_build/html
   - zip -r bicycleparameters-X.X.X.html.docs.zip *

9. Upload the basic data files if you've edited them.
