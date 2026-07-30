Computer Vision Project Summary

This project developed and evaluated a convolutional neural network (CNN) for handwritten digit classification using the MNIST dataset. The objective was to build an image classification model capable of accurately recognizing digits from grayscale images while demonstrating the complete deep learning workflow, including data preparation, model design, training, evaluation, and visualization.

The model was implemented using PyTorch Lightning, which streamlined the training process and enabled checkpointing, early stopping, and experiment logging. Model performance was assessed using training and validation loss, validation and test accuracy, and a confusion matrix to examine classification errors. Additional visualizations, including learning curves and examples of misclassified images, were used to communicate the model's performance and provide insight into the types of prediction errors that remained.

The final CNN achieved approximately 99.6% validation accuracy and 99.0% test accuracy, demonstrating excellent generalization to unseen handwritten digits. Most prediction errors occurred between visually similar digits, highlighting the inherent ambiguity present in some handwritten samples rather than limitations of the overall model. These results illustrate the effectiveness of convolutional neural networks for image classification tasks and emphasize the value of combining quantitative metrics with visual analysis to interpret model performance.


Metric:
Validation accuracy: 99.62%	
Validation loss: 0.0112	
Test accuracy: 99.01%	
Test loss: 0.0321

Misclassified example:

True: 5
Model: 6
Confidence: 73%

