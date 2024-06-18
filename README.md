Description:
This project demonstrates the implementation of a deep learning model for image classification. The model is designed to classify handwritten digits from the MNIST dataset. Key steps and components of the project include:

Data Augmentation:

Employed ImageDataGenerator from Keras to perform data augmentation, which enhances the model's robustness by generating new training samples through random transformations:
Rotation up to 40 degrees.
Width and height shifts up to 20%.
Shearing and zooming transformations.
Horizontal flipping.
Filling in new pixels with the nearest value.
Data Preparation:

Loaded the MNIST dataset, which consists of 60,000 training images and 10,000 validation images of handwritten digits.
Reshaped the data to include a single channel for grayscale images.
Data Visualization:

Visualized a sample image from the dataset to ensure correct loading and preprocessing.
Model Architecture:

Developed a Sequential CNN model comprising:
Two convolutional layers with ReLU activation and max-pooling.
Flattening layer to convert 2D matrices into a 1D vector.
Dense layer with ReLU activation.
Output layer with softmax activation for multi-class classification.
Model Compilation:

Compiled the model using Adam optimizer and sparse categorical cross-entropy loss function.
Set accuracy as the evaluation metric.
Model Training:

Trained the model using the augmented data generator for 30 epochs.
Validated the model on a separate validation set.
Model Evaluation:

Evaluated the model's performance on the validation dataset, achieving significant accuracy.
Displayed the model summary to provide insights into the architecture and parameter counts.
Performance Visualization:

Plotted training and validation loss over epochs to monitor convergence and detect overfitting.
Plotted training and validation accuracy over epochs to visualize the learning curve.
Classification Report:

Generated a classification report using Scikit-learn to provide detailed performance metrics such as precision, recall, and F1-score for each class.
Outcomes:

The project successfully implemented a CNN model capable of classifying handwritten digits with high accuracy.
Data augmentation proved to be effective in enhancing the model's generalization capabilities.
