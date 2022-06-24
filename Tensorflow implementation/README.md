# Tensorflow implementation

Using Tensorflow we tried to create a Convolutional Neural Network that could classify the breast tumor ultrasound images. We wanted to compaire our results with our previous models and see if we could achive better accuracy using a deep learning method.

Firstly, we imported the images and made a trainig set (402 images of tumor masks, 201 of each type) and a holdout set (10 masks of malignant tumors and 65 masks of benign tumors). We tried a lot of models but generally models with four 2D-Convolutional layers and a 16 node dense layer right before the output single node layer seemed to perform the best. Moreover, we added Dropout layers with 50% drop rate since anything less than that caused overfit.


![Screenshot_2](https://user-images.githubusercontent.com/61820986/175609986-60fe707b-4e7c-4151-8919-78d6f82627a8.png)


Since the 50% Drop Rate model seemed to respond well, we continued training it for another 100 epochs using Tensorflow's [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint?version=nightly) and [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) Callbacks. The trainig stopped after 15 since the validation loss could not be improved any further.
