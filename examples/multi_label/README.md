### Identify Image Hue Mulit-label Trainging Example

This sample will train a "Primary Color Detector".

There are three labels:

    0. RED
    1. GREEN
    2. BLUE

If a image's hue is close to red, it has label `RED`.
If a image's hue is close to yellow, it has label `RED` and `GREEN`.
If a image's hue is close to purple, it has label `RED` and `BLUE`.
and so on.

### Generate Trainging Image, Label, and Leveldb

    ./gen_training_data.py

will create following files:

- `hue_images` contain random generated images
- `train_file_label.txt` image file and label for training data
- `test_file_label.txt` image file and label for testing data
- `multi-label-train.leveldb` leveldb for training
- `multi-label-test.leveldb` leveldb for testing
- `mean.binaryproto` mean of training images

the content of `train_file_label.txt` looks like:

    ./hue_images/0.png -1 1 -1
    ./hue_images/1.png 1 1 -1
    ./hue_images/2.png -1 1 1
    ...

There is a `convert-multi-label-imageset.bin` tool. You can use it to convert your own training data.

### Run Training

    ./train.sh

will run the training process.

### Clean

    ./clean.sh

will clean all generated files.
