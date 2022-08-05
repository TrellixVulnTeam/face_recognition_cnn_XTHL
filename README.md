# face_recognition_cnn

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo">
</p>

## Description

AI to recognize from face images.

## Quickstart

To use the script, you need to install Python (at least 3.8 version).  
You also need to install some packages, you can find the list in the `requirements.txt` file or in the `setup.py` file.

You also need to install Mozilla Firefox to run the scripts. Please follow the instruction bellow to install Selenium:  
<https://selenium-python.readthedocs.io/installation.html>

To install them all automatically, type the following command at the root of the project :

```bash
pip install -r requirements.txt
```

You can also use the `setup.py` file, starting it with python will install all the python libs you need.

It will install all libs by typing the following command:

```bash
python setup.py install
```

or

```bash
pip install
```

## PyLint set up

The project is formatted via PyLint, if you want to check the project, you will need to install PyLin:

```bash
pip install pylint
```

> **Note**  
> If you use follow the steps in the `Quickstart` section then you wont need to install it again.

Then you can use it by typing the following command at the root of the project:

```bash
pylint .
```

It will scan all the project and print a report.

## Unit test scripts

The project is set up with some unit-test script to check the good behaviour of the project.

These check will load the last model and try to predict some image, if the prediction is wrong than the test failed and
signal the user preventing it from pushing to the other branches.

To use pytest, you can install it thought the command line:

```bash
pip install pytest
```

> **Note**  
> If you use follow the steps in the `Quickstart` section then you wont need to install it again.

## GitHub Actions

[![Python application](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml)

[![Pylint](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml)

[![CodeQL](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml)

The main branch will run some yaml script when you push or create a pull request to the main branch. It will verify the
behaviour of the code:

- The code is starting
- The code is formatted correctly
- The code quality

Once all tests are passed you can push or merge to the main branch.

## Documentation and Libraries

python:  
<https://www.python.org>

pylint:  
<https://pylint.pycqa.org/en/latest/>

pytest:  
<https://docs.pytest.org/en/stable/>

Scikit-Learn:  
<https://scikit-learn.org/stable/>

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/face_recognition_cnn)](https://github.com/Im-Rises/face_recognition_cnn/graphs/contributors)
