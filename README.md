### **Fruit Classification Model** 🍎🍊🍌

[![Fresh and Rotten Fruits](images/fruits.png)](images/fruits.png)

This repository contains the code for a deep learning model capable of classifying fruits as either fresh or rotten. The model was built using a combination of transfer learning, data augmentation, and fine-tuning to achieve high accuracy.

***

### **1. Project Goal**

The goal of this project is to develop an automated solution for fruit quality control. The model is trained to recognize and classify images of fresh and rotten fruits, which can be applied to business problems such as sorting in production lines or monitoring food inventory to reduce waste.

***

### **2. Dataset**

The model was trained on the "Fruits Fresh and Rotten for Classification" dataset from Kaggle. The dataset is organized into six distinct categories, as illustrated in the image above:

* Fresh apples
* Fresh oranges
* Fresh bananas
* Rotten apples
* Rotten oranges
* Rotten bananas

***

### **3. Methodology**

The model architecture is based on **transfer learning** from a **VGG16 model** pre-trained on the ImageNet dataset. This approach leverages the powerful feature extraction capabilities of a model trained on a large and diverse image dataset.

The key steps in the methodology were:

* **Freezing Layers**: The initial layers of the VGG16 model were frozen to prevent their weights from being updated during the initial training phase. This preserves the general, low-level features learned from ImageNet.
* **Data Augmentation**: To improve the model's generalization and prevent overfitting, the training data was augmented with various transformations. These included random horizontal flips, rotations, and color jitter.
* **Fine-Tuning**: After initial training, the entire model was unfrozen, and a very low learning rate was used to fine-tune all layers. This allows the pre-trained model's weights to be slightly adjusted to better fit the specific fruit dataset.

***

### **4. Results**

The model achieved a final **validation accuracy of 92%**. This high accuracy demonstrates that the model is highly effective at distinguishing between fresh and rotten fruits across different categories.

***

### **5. Future Work**

Potential improvements and future work for this project could include:

* **Exploring different architectures**: Experimenting with other pre-trained models, such as ResNet or EfficientNet, which may offer better performance or be more computationally efficient.
* **Expanding the dataset**: Adding more fruit varieties to the dataset and including images with different lighting conditions, backgrounds, or angles to make the model more robust in real-world scenarios.
