# Speech_Emotion_Recognition

The goal of this project is to develop an innovative mobile app capable of accurately recognizing emotional states by addressing acoustic variance between genders. We aim to achieve this by utilizing a gender-specific dataset and employing advanced machine learning and deep learning techniques.

### Dataset
We have assembled a comprehensive dataset that includes recordings from both male and female speakers. The SLR45 Database will serve as the foundation for gender identification using a Gaussian Mixture Model (GMM). Additionally, we have integrated several well-known emotion-specific datasets, namely RAVDESS, CREMA-D, SAVEE, and TESS, to train and validate our emotion recognition model.

## Gender Dataset
![Picture10](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/156553ac-32f4-4585-b448-5c5aa9fbce71)

## Emotion Dataset
![Picture9](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/34820e01-5807-419f-a8f3-7ba9f15c2d4a)

###  Architecture
![Picture1](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/d99c0412-84f7-4af7-89d9-29cadad464f9)
### Feature Extraction
To capture essential information from the speech signals, we leverage Mel Frequency Cepstral Coefficients (MFCC) to extract relevant features like pitch, intonation, and prosody. This step is critical in ensuring that our model can effectively distinguish emotional cues unique to different genders.

### Gender Identification
The first stage of our emotion recognition pipeline involves accurately identifying the gender of the speaker. To achieve this, we employ a Gaussian Mixture Model (GMM), trained on the SLR45 Database, which allows us to reliably discern between male and female speakers.
### Gender Model
![Picture11](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/65e9ab63-337f-4b3d-b384-01eceb614b36)

### Emotion Recognition
In the second stage, we utilize deep learning classifiers, specifically Convolutional Neural Networks (CNNs), to perform emotion recognition. Our ensemble model is trained on a combination of RAVDESS, CREMA-D, SAVEE, and TESS datasets, providing it with a diverse range of emotional expressions for robust and accurate predictions.

### Emotion Model
![Picture12](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/2b43882f-263e-493c-8d0d-739f10013708)

### App Development
The final deliverable of this project is an intuitive and user-friendly mobile application that takes in speech input and accurately predicts the speaker's emotional state. The app will be developed for both Android and iOS platforms, allowing widespread accessibility and usage.

### Evaluation and Performance Metrics
Here we have used accuracy, confusion matrix, precision, and recall to evaluate the performance of our emotion recognition model. Accuracy will provide an overall measure of the model's correctness, while the confusion matrix will offer a detailed breakdown of its predictions for each emotion and gender class. Precision will assess the accuracy of positive predictions, while recall will measure the model's ability to identify all relevant emotions and genders effectively.  Through rigorous optimization and fine-tuning, we aim to create an accurate and robust emotion recognition app that considers gender-specific acoustic variance, contributing to a better understanding of emotions in various real-world scenarios. 






#### Experimental Results
![Picture5](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/33b43df3-1bc4-4ad4-9661-ce682c91e907)

![Picture6](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/b228befa-330f-49d3-8337-f09443979c44)



#### Female Model
![Picture7](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/a9abb677-9fff-4356-bf88-f8b6d11ddbd5)
![Picture13](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/3597be6d-8328-4f76-94cb-f7b06a5701f0)


#### Male Model
![Picture8](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/aa45f2b9-8fbc-4e19-b958-4d814b76dbec)
![Picture14](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/0e69cc3a-1149-4de5-99e7-824f0cb685c7)



### Output
![Picture2](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/dfaf47d5-8167-48f4-8e7f-296b5487bbd9)
![Picture3](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/359580b2-faaf-44a0-bb64-538b09bf6df4)
![Picture4](https://github.com/Gunti-Swathi/Gender_Specific_Speech_Emotion_Recognition/assets/75379302/af28264a-1b14-4843-afa3-a77c8a4ca357)


