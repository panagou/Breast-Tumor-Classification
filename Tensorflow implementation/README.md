# Tensorflow implementation

Using Tensorflow we tried to create a Convolutional Neural Network that could classify the breast tumor ultrasound images. We wanted to compaire our results with our previous models and see if we could achive better accuracy using a deep learning method.

Firstly, we imported the images and made a trainig set (402 images of tumor masks, 201 of each type) and a holdout set (10 masks of malignant tumors and 65 masks of benign tumors). We tried a lot of models but generally models with four 2D-Convolutional layers and a 16 node dense layer right before the output single node layer seemed to perform the best. Moreover, we added Dropout layers with 50% drop rate since anything less than that caused overfit. 

We trained the models using 10-fold cross validation and a data batch size of 32 for 100 epochs.

![Screenshot_3](https://user-images.githubusercontent.com/61820986/175614011-e6cedf3e-caf3-4f80-809e-a6a2ebbd3d19.png)

![Screenshot_2](https://user-images.githubusercontent.com/61820986/175609986-60fe707b-4e7c-4151-8919-78d6f82627a8.png)


Since the 50% Drop Rate model seemed to respond well, we continued training it for another 100 epochs using Tensorflow's [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint?version=nightly) and [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) Callbacks. Again we used 10-fold cross validation with a batch size of 32 . The trainig stopped after 15 epochs since the validation loss could not be improved any further.

![Screenshot_4](https://user-images.githubusercontent.com/61820986/175616039-f21a3880-a08f-4a11-b478-d884991ff7b5.png)

![Screenshot_2](https://user-images.githubusercontent.com/61820986/175616452-1522cb68-736b-4ad1-9f9e-7e6e5dd12a26.png)

We can see that the extra training slightly improved the model but not by much. Now we are ready to test the models on our holdout set. We tested the 50% drop rate model before and after the extra training as well as the 10% drop rate model.


<p align="center">
  <img  src="https://user-images.githubusercontent.com/61820986/175645026-947047b0-7fe6-4d61-a21f-bdeec1620efe.png">
  <br>
  <br>
  <img  src="https://user-images.githubusercontent.com/61820986/175644985-da41f7ac-1c89-41ac-9cf3-4ed2750dd4e1.png">
</p>


Here we see that the ROC curves of the two 50% models are almost identical, as well as the AUC score. That largly depends on the way we chose the Holddout Set as well as the images that it contains. Maybe if we added more ultrasound images of Malignant tumors the results would differ.

Nonetheless, we can conlude that the three CNNs models we tested are fairly accurate. The 50% drop rate models scored an AUC value of 96.6% and a True Negative Rate value of 73.85% while maintaining a 0% False Negative Rate. Although less accurate than the Logistic Regression and the XGBClassifier, they are more accurate than the rest of the Classifiers we implemented.
