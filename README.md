# Number Plate Recognition with TensorFlow

This repository is going to implement a simple number plate recognition using CNN model with TensorFlow. Please notice that this program can **only** recognize the pattern of the number plate as follow:

![](https://i.imgur.com/IQdsTrX.jpg)

> More information about [number plate recognition with Tensorflow](http://matthewearl.github.io/2016/05/06/cnn-anpr/)

---
## File Structure

```bash
NumberPlate_Recognition     # This is ./ in this repository
|--- input/                 # Input image of number plate
|--- output/                # Output image of number plate's recognition
|--- common.py
|--- detect.py
|--- model.py
```

---
## Prerequisite

* Before executing, you need to install the following packages
    * Install **OpenCV** using `pip`
        ```bash
        $ [sudo] pip install opencv-python
        ```
    * Install **TensorFlow** using `pip`
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

* Put the image of number plate in the folder `./input/`.
* Execution
    ```bash
    # Make sure your current directory is "./src/"
    $ python detect.py t1.jpg weights.npz
    $ python detect.py t2.jpg weights.npz
    ```
* Open the folder `./out/` to see the result of recognition. Notice that the filename of the result will same as the input.
    ```bash
    # List the recognition result
    $ cd ../out && ls
    t1.jpg  t2.jpg
    ```

---
## References

* [TensorFlow](https://www.tensorflow.org/)
* [opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)
* [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)

---
## Author

* [David Lu](https://github.com/yungshenglu)

---
## License

[GNU GENERAL PUBLIC LICENSE Version 3](LICENSE)