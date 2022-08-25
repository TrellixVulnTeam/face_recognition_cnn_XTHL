# face_recognition_cnn

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/185226532-1378b39e-210d-4400-a4a1-a979572ed655.png" alt="skeletonLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/185226526-8cb9c3b2-7d1a-41b5-ba1e-50ba1f5b391e.png" alt="tensorflowLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/172961027-fd9185a5-da77-46e3-97b1-54e99e242822.png" alt="opencvLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/186103062-770f199a-b55a-43a5-ab43-0cd1685cb0dd.png" alt="tkinterLogo" style="height:50px;">
</p>

## Description

AI to recognize from face images. It is a convolutional neural network (CNN) based face recognition system.
The project is split in two parts, one using the script I found in the Sci-kit learn documentation, I modified it to try
to get the best result possible.  
I also implemented a CNN using Resnet50 and transfer learning, to try to get the best result possible.
The app use the ResNet50 model trained with LFW dataset.

The main scripts are in the `src` folder, the `test` folder contains the test scripts and the notebooks are in
the `notebooks` folder.

Datasets:

- LFW

[//]: # (- Large-scale CelebFaces Attributes)

[//]: # (- IMDB-WIKI)

[//]: # (- UMD-Faces)

[//]: # (- VGGFace2)

## Images

### UI screenshot

![ui_image](https://user-images.githubusercontent.com/59691442/186635140-9a48545e-4089-4e50-a062-b73c160529dc.png)

## Quickstart

To use the script, you need to install Python (at least 3.8 version).  
You also need to install some packages, you can find the list in the `requirements.txt` file or in the `setup.py` file.

To install them all automatically, type the following command at the root of the project:

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

Once everything is installed, you can open the script `main.py` in the `src` folder.
It will create a window and start your camera.

There is two ways to use the app:

- Open an image by clicking on the `Open image` button.
- Use the webcam by clicking on the `Find face` button.

If you have selected an image from your computer, the image will automatically find the face in your image, crop and
display it in the left part of the app.
You can then press the `predict` button to check who it is.

If you use the camera from your computer, when you clicked find face, your face will be croped from the video and
display in the left of the app, then you can press the `predict` button to check who you look like.

### Example image selection

Start the `main.py` script file, once the UI is open, click on the `Open image` button and select an image button,
select your image (in my case Arnold Schwarzenegger).

<p align="center">
  <img src="https://user-images.githubusercontent.com/59691442/186625931-93c1eb04-3cb6-4cbd-b065-7482fbef4fef.jpg" alt="arnoldImage" style="height:300px;">
</p>

Once selected, the image will be displayed in the left part of the app with the face croped from the image.

You can then press the `predict` button to check who it is.

![arnold predicted](https://user-images.githubusercontent.com/59691442/186628360-9e153db8-ee27-4026-bda2-be9d68e8f86a.png)

[//]: # (![ui_example]&#40;https://user-images.githubusercontent.com/59691442/186625879-1b30fd3a-2968-4a6e-bccb-2726249115e7.png&#41;)

Arnold Schwarzenegger is the predicted person.

### Example image from webcam

Start the `main.py` script file, once the UI is open, click the `ON/OFF` button to enable the camera, then press
the `Find face` button. Your face croped from the video will be displayed in the right upper corner of the app.
You can then press the `predict` button to check who you look like.

![me_croped](https://user-images.githubusercontent.com/59691442/186629337-f8fb1f26-6634-4658-a7bc-37ecac7fa194.png)

> **Note**  
> You can find a list of outputs in the `Datasets results` section

## Project architecture

~~~
face_recognition_cnn
├── .github
|  ├── labeler.yml
|  ├── release.yml
|  |  |── black.yml
|  |  |── codeql-analysis.yml
|  |  |── dependency-review.yml
|  |  |── greetings.yml
|  |  |── label.yml
|  |  |── pylint.yml
|  |  |── python-app.yml
|  |  |── stale.yml
├── datasets
|  ├── dataset-here.text
|  ├── lfw
|  ├── remake_dataset.py
├── face_detection_weights
|  ├── download-link.txt
|  ├── haarcascade_frontalface_default.xml
├── models
├── resnet50_dl_lfw
├── resnet50_dl_lfw_empty
├── notebook
|  ├── dl_lfw.ipynb
|  ├── ml_lfw.ipynb
├── src
|  ├── main.py
|  ├── person_dictionary.py
|  ├── training.py
|  ├── user_interface.py
├── weights
|  ├── resnet50_lfw.h5
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pylintrc
├── README.md
├── requirements.txt
├── setup.py
~~~

## Training

### How to train

To train the model, you need to download the LFW (Labeled Faces in the Wild) dataset and put it in the `datasets`
folder.
<http://vis-www.cs.umass.edu/lfw/>

Depending on the minimum number of images a person need to have you
need to start the `remake_dataset.py` script.
It will untar the dataset and delete all persons who don't have at least the minimum number of images (you can change
this parameter by changing the variable `MINIMUM_IMAGES_BY_CLASS` in the script).

The weights and the model will be respectively saved in the `weights` and `models` folder.

### Dataset images

![myplot4](https://user-images.githubusercontent.com/59691442/186006740-c3bf2f78-a252-439e-ad11-0db503f0c35f.png)

### Dataset results

#### LFW

![myplot](https://user-images.githubusercontent.com/59691442/186006713-7d9eedda-f51a-43ed-8492-87449dc72fcc.png)

LFW 70 persons minimum:

- SVM and Eigen-faces:
    - Accuracy: 0.85

- Resnet50:
    - Accuracy: 0.96

LFW 10 persons minimum:

- SVM and Eigen-faces:
    - Accuracy: 0.44

- Resnet50:
    - Accuracy: 0.78

> **Note**  
> The accuracy is shown is from the test par of the datasets.

<!--
### Large-scale CelebFaces Attributes
### IMDB-WIKI

- Resnet50:
    - Accuracy: 0.62
    - Precision: 0.9)
    - Recall: 0.9)
    - F1 score: 0.9)
-->

#### List of outputs

<details>
  <summary>List of predictable persons</summary>
  <p>
    Abdullah_Gul,
    Adrien_Brody,
    Alejandro_Toledo,
    Alvaro_Uribe,
    Amelie_Mauresmo,
    Andre_Agassi,
    Andy_Roddick,
    Angelina_Jolie,
    Ann_Veneman,
    Anna_Kournikova,
    Ari_Fleischer,
    Ariel_Sharon,
    Arnold_Schwarzenegger,
    Atal_Bihari_Vajpayee,
    Bill_Clinton,
    Bill_Gates,
    Bill_McBride,
    Bill_Simon,
    Britney_Spears,
    Carlos_Menem,
    Carlos_Moya,
    Catherine_Zeta-Jones,
    Charles_Moose,
    Colin_Powell,
    Condoleezza_Rice,
    David_Beckham,
    David_Nalbandian,
    Dick_Cheney,
    Dominique_de_Villepin,
    Donald_Rumsfeld,
    Edmund_Stoiber,
    Eduardo_Duhalde,
    Fidel_Castro,
    George_HW_Bush,
    George_Robertson,
    George_W_Bush,
    Gerhard_Schroeder,
    Gloria_Macapagal_Arroyo,
    Gonzalo_Sanchez_de_Lozada,
    Gordon_Brown,
    Gray_Davis,
    Guillermo_Coria,
    Halle_Berry,
    Hamid_Karzai,
    Hans_Blix,
    Harrison_Ford,
    Hillary_Clinton,
    Howard_Dean,
    Hu_Jintao,
    Hugo_Chavez,
    Ian_Thorpe,
    Igor_Ivanov,
    Jack_Straw,
    Jackie_Chan,
    Jacques_Chirac,
    Jacques_Rogge,
    James_Blake,
    James_Kelly,
    Jason_Kidd,
    Javier_Solana,
    Jean-David_Levitte,
    Jean_Charest,
    Jean_Chretien,
    Jeb_Bush,
    Jennifer_Aniston,
    Jennifer_Capriati,
    Jennifer_Garner,
    Jennifer_Lopez,
    Jeremy_Greenstock,
    Jiang_Zemin,
    Jiri_Novak,
    Joe_Lieberman,
    John_Allen_Muhammad,
    John_Ashcroft,
    John_Bolton,
    John_Howard,
    John_Kerry,
    John_Negroponte,
    John_Paul_II,
    John_Snow,
    Joschka_Fischer,
    Jose_Maria_Aznar,
    Juan_Carlos_Ferrero,
    Julianne_Moore,
    Julie_Gerberding,
    Junichiro_Koizumi,
    Keanu_Reeves,
    Kim_Clijsters,
    Kim_Ryong-sung,
    Kofi_Annan,
    Lance_Armstrong,
    Laura_Bush,
    Lindsay_Davenport,
    Lleyton_Hewitt,
    Lucio_Gutierrez,
    Luiz_Inacio_Lula_da_Silva,
    Mahathir_Mohamad,
    Mahmoud_Abbas,
    Mark_Philippoussis,
    Megawati_Sukarnoputri,
    Meryl_Streep,
    Michael_Bloomberg,
    Michael_Jackson,
    Michael_Schumacher,
    Mike_Weir,
    Mohammad_Khatami,
    Mohammed_Al-Douri,
    Muhammad_Ali,
    Nancy_Pelosi,
    Naomi_Watts,
    Nestor_Kirchner,
    Nicanor_Duarte_Frutos,
    Nicole_Kidman,
    Norah_Jones,
    Paradorn_Srichaphan,
    Paul_Bremer,
    Paul_Burrell,
    Paul_Wolfowitz,
    Pervez_Musharraf,
    Pete_Sampras,
    Pierce_Brosnan,
    Queen_Elizabeth_II,
    Recep_Tayyip_Erdogan,
    Renee_Zellweger,
    Ricardo_Lagos,
    Richard_Gephardt,
    Richard_Gere,
    Richard_Myers,
    Roger_Federer,
    Roh_Moo-hyun,
    Rubens_Barrichello,
    Rudolph_Giuliani,
    Saddam_Hussein,
    Salma_Hayek,
    Serena_Williams,
    Sergey_Lavrov,
    Sergio_Vieira_De_Mello,
    Silvio_Berlusconi,
    Spencer_Abraham,
    Taha_Yassin_Ramadan,
    Tang_Jiaxuan,
    Tiger_Woods,
    Tim_Henman,
    Tom_Cruise,
    Tom_Daschle,
    Tom_Hanks,
    Tom_Ridge,
    Tommy_Franks,
    Tommy_Thompson,
    Tony_Blair,
    Trent_Lott,
    Venus_Williams,
    Vicente_Fox,
    Vladimir_Putin,
    Walter_Mondale,
    Wen_Jiabao,
    Winona_Ryder,
    Yoriko_Kawaguchi,
  </p>
</details>

## Unit test scripts

The project is set up with some unit-test script to check the good behaviour of the project.

These check will load the last model and try to predict some image, if the prediction is wronger than the test failed
and
signal the user preventing it from pushing to the other branches.

To use pytest, you can install it thought the command line:

```bash
pip install pytest
```

You can then run pytest by writing the following command at the root of the project:

```bash
pytest
```

> **Note**  
> If you followed the steps in the `Quickstart` section then you won't need to install it again.

## PyLint set up

The project is formatted via PyLint, if you want to check the project, you will need to install PyLin:

```bash
pip install pylint
```

Then you can use it by typing the following command at the root of the project:

```bash
pylint .
```

It will scan all the project and print a report.

> **Note**  
> If you followe the steps in the `Quickstart` section then you won't need to install it again.

## Code formatter

To format the code, the project is set up with a `.pylintrc` that change some rules about how the code should be.
You can install black by typing

```bash
pip install black
```

and install the jupyter extension to black to format them too.

```bash
 pip install 'black[jupyter]'
```

or if you followed the instruction in the `Quickstart` section and install the requirements by using
the `requirements.txt` or the `setup.py` file, you can already use it.
To start a check and correct the code type:

```bash
black --check --target-version=py35 .
```

> **Warning**  
> To format the Jupyter Notebook file you need to install it through the command line, it won't be installed with
> the `requirements.txt` and `setup.py`.

## Git Large File Storage

To store online the weights of the model, I use the Git Large File Storage app which I specified in the `.gittatributes`
file which file to store in the git repository.

Git Large File Storage:  
<https://git-lfs.github.com>

To add a new large file to the repository, type the following command:

```bash
git lfs track <file_path>
```

It will also register it in the `.gitattributes` file.

> **Warning**  
> Al files registered for LFS in the `.gitattributes` file will be stored in the as only one entity (no versioning).
> Originally all files in the weights folder will be stored online.

## GitHub Actions

[![Python application](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/python-app.yml)
[![Pylint](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/pylint.yml)
[![Black code-style](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/black.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/black.yml)
[![CodeQL](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Im-Rises/face_recognition_cnn/actions/workflows/codeql-analysis.yml)

The main branch will run some yaml script when you push or create a pull request to the main branch. It will verify the
behaviour of the code:

- Python application : The script will run the application and check if the prediction is correct.
- Pylint : The script will run pylint and check if the code respect some norms.
- Black : The script will run black and check if the code is formatted.
- CodeQL : The script will run codeql and check if the code is clean.

Once all tests are passed you can push or merge to the main branch.

## Documentation and Libraries

Python:  
<https://www.python.org>

Pylint:  
<https://pylint.pycqa.org/en/latest/>

PyLintRc File:  
<https://github.com/PyCQA/pylint/blob/main/pylintrc>

Black Formatter:  
<https://github.com/psf/black>

Pytest:  
<https://docs.pytest.org/en/stable/>

Scikit-Learn:  
<https://scikit-learn.org/stable/>
<https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>

OpenCv:  
<https://opencv.org>

OpenCv Face Detections weights:
<https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml>

Tensorflow:  
<https://www.tensorflow.org>
<https://www.tensorflow.org/tutorials/load_data/images>  
<https://www.tensorflow.org/tutorials/images/data_augmentation>  
<https://www.tensorflow.org/guide/keras/preprocessing_layers>
<https://www.tensorflow.org/tutorials/images/transfer_learning>

Datacorner:  
<https://datacorner.fr/category/ia/deep-learning/>
<https://datacorner.fr/vgg-transfer-learning/>

Git Large FIle Storage:  
<https://git-lfs.github.com>

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
