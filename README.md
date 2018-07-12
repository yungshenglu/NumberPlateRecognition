# Number Plate Recognition

This repository is some pactice of data parsing by using Python. Notice that this repository is lab in "Workshop on AI & Big Data Analytics 2018".

> * [Number plate recognition with Tensorflow](http://matthewearl.github.io/2016/05/06/cnn-anpr/)

---
## Prerequisite

* Before executing, you need to install the following packages
    * Install **OpenCV** packages on Python
        ```bash
        $ [sudo] pip install opencv-python
        ```
    * Install **TensorFlow** packages on Python
        ```bash
        $ [sudo] pip install -U tensorflow
        ``` 
* When installing **TensorFlow**, you may meet the following problems. However, if install successfully, you can follow the execution
    * Error message: `launchpadlib 1.10.3 requires testresources, which is not installed.`
        ```bash
        $ [sudo] pip install launchpadlib
        ```
    * Error message: `Cannot uninstall 'enum34'. ...`
        ```bash
        $ sudo apt-get remove python-enum34
        ```

---
## Execution

* Put the image of number plate in folder `input/`.
* Execute
    ```bash
    # Execute number plate recognition
    $ python detect.py t1.jpg weights.npz
    $ python detect.py t2.jpg weights.npz
    ```
* Open the folder `./output/` to see the result of recognition. Notice that the filename of the result will same as the input.
    ```bash
    # List the recognition result
    $ cd ./output && ls
    t1.jpg  t2.jpg
    ```

---
## Framework

* `detect.py`
* `model.py`
* `common.py`
* `input/` - Input image of number plate
* `output/` - Output image of number plate's recognition

---
## References

* [TensorFlow](https://www.tensorflow.org/)
* [opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)

---
## Author

* [David Lu](https://github.com/yungshenglu)