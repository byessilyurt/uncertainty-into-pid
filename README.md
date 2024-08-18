
# Introduction

The objective of this research is to develop and evaluate deep learning models for Pedestrian Intention Detection (PID) in urban traffic scenarios. The project explores the integration of uncertainty modeling, specifically aleatoric and epistemic uncertainties, to enhance the performance and robustness of PID systems.

# Dataset

The project uses the Pedestrian Intention Estimation (PIE) dataset, a large-scale dataset designed for studying pedestrian behavior in urban environments. The dataset includes video sequences, annotations, and contextual information.

More details about the PIE dataset can be found in the publication:


`@article{rasouli2019pie,
    author = "Amir Rasouli, Iuliia Kotseruba, John K. Tsotsos",
    title = "{Pie: A large-scale dataset and models for pedestrian intention estimation and trajectory prediction}",
    journal = "Proceedings of the IEEE/CVF International Conference on Computer Vision",
    year = "2019",
}` 

The `pie_data.py` file provides an interface for accessing and manipulating the PIE dataset, including functions for extracting video frames, generating annotations, and preparing the data for analysis. The script is available at [PIE Dataset GitHub Repository](https://github.com/aras62/PIE/).

# Model Architectures

Three deep learning architectures were implemented and evaluated in this research:

-   **ResNet50**: A deep residual network that mitigates the vanishing gradient problem, allowing for more effective training of deep models.
-   **VGG16**: A simple and effective convolutional neural network with a deep stack of convolutional layers.
-   **AlexNet**: A classic architecture known for its efficiency in image classification tasks.

Each model was pretrained on the ImageNet dataset and fine-tuned on the PIE dataset for pedestrian intention detection.
