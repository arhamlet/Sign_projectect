# Sign-Language Digits Image Classification
The sign-language dataset was created and then classification was performed on it.

## Dataset
### Preview
<img src="signlanguagedigits.png">

The dataset contains ten classes ('0' to '9'). It was resized to 224 x 224 for the googlenet. A csv file with the image labels is also provided.

## Training
The dataset was trained using Googlenet. The optimizer used was Adam and the loss function was CrossEntropy loss. The dataset was trained with a learning rate of 0.001. Learning rate scheduler was also used to improve the accuracy. 
The dataset was divided into training set, validation set and the test set. After training, 99 percent accuracy was achieved. 

## How to Run?
Keep the dataset in the images folder and run the train.py file.