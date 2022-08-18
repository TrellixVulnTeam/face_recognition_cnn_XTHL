# face_recognition_cnn

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/185226532-1378b39e-210d-4400-a4a1-a979572ed655.png" alt="skeletonLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/185226526-8cb9c3b2-7d1a-41b5-ba1e-50ba1f5b391e.png" alt="tensorflowLogo" style="height:50px;">
</p>

## Description

AI to recognize from face images. It is a convolutional neural network (CNN) based face recognition system.
The project is split in two parts, one using the script I found in the Sci-kit learn documentation, I modified it to try
to get the best result possible.
I also implemented a CNN using Resnet50 and transfer learning, to try to get the best result possible.

- Resnet50:
    - Accuracy: 0.9
    - Precision: 0.9
    - Recall: 0.9
    - F1 score: 0.9
- SVM:
    - Accuracy: 0.9
    - Precision: 0.9
    - Recall: 0.9
    - F1 score: 0.9

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

## Project architecture

<!--
~~~
project-revision
├── dataset
|  ├── test
|  ├── train
├── docs
├── misc
|   ├── dataset
|   ├── resize.py
├── src
|   ├── denoise
|   |   |── __init__.py
|   |   |── denoise.py
|   |   |── errors.py
|   ├── noise
|   |   |── __init__.py
|   |   |── errors.py
|   |   |── noise.py
|   ├── results
|   ├── cobra.py
|   ├── cobramachine.py
|   ├── denoise.py
|   ├── helper.py
|   ├── main.py
|   ├── req.txt
├── README.md
├── guidelines.md
├── proposal.md
~~~
-->

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

You can then run pytest by writing the following command at the root of the project:

```bash
pytest
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

## Code formatter

To format the code, the project is set up with a `.pylintrc` that change some rules about how the code should be.
You can install black by typing

```bash
pip install black
```

or if you followed the instruction in the `Quickstart` section and install the requirements by using
the `requirements.txt` or the `setup.py` file, you can already use it.
To start a check and correct the code type:

```bash
black --check --target-version=py35 .
```

## GitHub Actions

[![Python application](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml)
[![Pylint](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml)
[![CodeQL](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml)

The main branch will run some yaml script when you push or create a pull request to the main branch. It will verify the
behaviour of the code:

- Python application : The script will run the application and check if the prediction is correct.
- Pylint : The script will run pylint and check if the code is formatted.
- CodeQL : The script will run codeql and check if the code is clean.

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

Scikit-Learn ML Face Recognition script:  
<https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>

Tensorflow:  
<https://www.tensorflow.org>

Tensorflow transfer learning:  
<https://www.tensorflow.org/tutorials/images/transfer_learning>

PyLintRc File:  
<https://learn.adafruit.com/improve-your-code-with-pylint/pylintrc>

Black Formatter:
<https://github.com/psf/black>

Datacorner transfer learning:  
<https://datacorner.fr/vgg-transfer-learning/>

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/face_recognition_cnn)](https://github.com/Im-Rises/face_recognition_cnn/graphs/contributors)

<!--
<https://l.messenger.com/l.php?u=https%3A%2F%2Ftowardsdatascience.com%2Fwhich-celebrity-are-you-d8c6507f21c9&h=AT2JXOiux7hfqK4Gey9gcOx_xiuX1g3oIkwaiiJ_R4_OPPzaZSGZNU9ta7Iu--xWiCKV9fb4UXF7pF2QQEekJCCDo-wAY0vBqUxFdPpKBvvxpYqFLdLNLf6OQXz80J6VdMI9oA>

https://towardsdatascience.com/which-celebrity-are-you-d8c6507f21c9

https://sefiks.com/2019/05/05/celebrity-look-alike-face-recognition-with-deep-learning-in-keras/

https://medium.com/analytics-vidhya/celebrity-recognition-using-vggface-and-annoy-363c5df31f1e

https://github.com/serengil/tensorflow-101
-->
