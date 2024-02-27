# deep-learning-challenge
MSU Data Analytics BootCamp Module 21 Challenge

Notebook/code file names: 'AlphabetSoupCharity_InitialTrainTest.ipynb' and 'AlphabetSoupCharity_Optimization.ipynb' located in the root repository directory

# Title

**Grant Applicant Success Classification - Using A Neural Network to Model Grant Applicant Success Based on Previous Applicant Data**

# Overview and Goal of the Analysis

The goal of this analysis was to use a neural network to create a classification model of future grant funding success for an organization that distributes competitive grants. Here we employed a supervised learning approach built on previous awarded grants and their predetermined outcomes. Specifically, we took applicant supplied data (feature variables) collected from previous funding applications and the ultimate outcome of their funding (target variable: a binary successful or unsuccessful result) to feed into a neural network and train a binary classification model. After the model was trained and the “best” model was found (for specific parameters trained on), it was evaluated for accuracy with reserved test data derived from the original dataset. Data preprocessing and model hyperparameters were then tuned and optimized to improve the classification model accuracy rate to at least 75%. Once a high-performing model is found, future grant applicant data can then be fed into a deployed model to assess the likelihood of ultimate applicant success and help guide funding decisions for the organization.

**Results

# Data Preprocessing

- Target variable name: IS_SUCCESSFUL. In this case, the variable indicates if a particular grant applicant was able fulfill the goal outlined in their application to a level deemed by the organization as successful. We are attempting to use the feature variables (listed below) to discover the best classification model that will predict this ‘IS_SUCCESSFUL’ target variable as a binary yes or no.
- Feature variable names: Initially, our feature variables include ‘AFFILIATION’ (affiliated sector of industry), ‘CLASSIFICATION’ (government organization classification), ‘USE_CASE’ (use case for funding), ‘ORGANIZATION’ (organization type) ‘STATUS’ (active status), ‘INCOME_AMT’ (income classification), ‘SPECIAL_CONSIDERATIONS’ (special considerations for application), ‘ASK_AMT’ (funding amount requested). Each of these variables were used to train the initial model and evaluate their usefulness.
-Two feature variables not mentioned above ‘EIN’ and ‘NAME’ were removed before initial model training, since they contained no model pertinent information. Two other variables: ‘STATUS’ and ‘SPECIAL_CONSIDERATIONS’ were also removed to fine tune the model training, since they also appeared to contain little model pertinent information and could possibly lead to noise during the model training that prevents finding the optimal model.

# Compiling, Training, and Evaluating the Initial Model

-The initial model (contained in ‘AlphabetSoupCharity_InitialTrainTest.ipynb’) was trained using a one hidden layer neural net, with an input layer of 80 units using the ReLU activation function. The only hidden layer was 30 units also using ReLU, and the output layer was 1 unit with a sigmoid activation function. The loss function used was binary_crossentropy with the Adam optimizer and the default learning rate (and other parameters). These parameters were chosen for several reasons, including: computational efficiency due to the relative lack of complexity for our dataset (don’t need many layers or units for this fairly simple dataset), activation function ReLU has robustness to many types of input data (the sigmoid output layer activation function and the loss function were specifically used because we are attempting to solve a binary classification problem), an acceptable balance between model over- and underfitting with a low complexity dataset (a relatively small number of layers and units should be adequate)
-Unfortunately, we were unable to reach the overall target accuracy of 75% with our initial neural net parameters. Using our initial model training parameters, we did reach an overall model accuracy of approximately 72.3% with a training loss of 57.5%, and a validation accuracy of no better than 74%.
-In order to improve model performance, several additional data preprocessing and hyperparameter optimization techniques were employed (as seen in ‘AlphabetSoupCharity_Optimization.ipynb’). For our initial model, basic preprocessing such as one-hot encoding, categorical bucketing/binning, and feature variable scaling and transform were applied to the dataset. All the above steps were again performed to fine tune our model, but with several other additional steps. These included:
 1. Data columns that were unlikely to provide any model improving information were dropped from the dataset 
2. Bucketed/binned categorical data had the cutoff value reduced slightly to include a bit more un-bucketed/un-binned data for training and perhaps reveal information that improves the model
3. Class imbalances in the target variable were addressed with synthetic minority oversampling (SMOTE)
4. Additional hidden layers were included in the neural network with a greater number of units per layer
5. Dropout was applied to the neural net layers to help mitigate any model overfitting
6. A learning rate schedule was implemented to help tune the learning rate for the optimizer during model training 
When these methods failed to appreciably improve the model, hyperparameter optimization using Hyperband was implemented to find the best combination of hyperparameters to train the model. Optimization parameters included layer activation function, layer number, unit number, and learning rate.

**Summary

Even with the additional data preprocessing and Hyperband hyperparameter optimization, we were unable to reach an overall accuracy of 75% or greater. After implementing the Hyperband optimization, we were able to improve overall model accuracy very slightly to roughly 72.5% with a training loss of 56.1%, and a validation accuracy of around 74.7%. The best performing model hyperparameters are found in ‘AlphabetSoupCharity_Optimization.ipynb’ and the overall best performing model was output to a .h5 file. 
This may be a case where a neural net is not the best approach to model training for this dataset. A perhaps more appropriate approach to training our binary classification model might be a decision tree-based approach. There are serval reasons for this including the relatively small size of the dataset (neural nets perform best with very large amounts of data), the more reasonable feature engineering requirements of decision trees (decision trees are less sensitive to data scale and data types than neural nets), and the less “black-box” nature of how a decision tree model works, especially in the case where we are attempting to train a binary classification model that ultimately may help decide who receives large sums of money. Here, it is important that if a model is used to help decide where grants are awarded, it is paramount that it is clear how the model arrived at that decision. For example, it may allow the organization to provide clear and useful feedback to rejected applicants for future re-application or avoid any implications of impropriety in who was given awards.

*Code sourcing statement*
-----------------------

I did use a natural language description of some desired code functions entered into ChatGPT 3.5 to help build the code structure. I did copy pieces of that code to be more efficient, but I tailored it to fit all the desired functions of this particular project. I did not directly copy and paste any of this code from the internet otherwise (e.g., from StackExchange or any other webpage). I did not seek any assistance or use code written by my peers or instructors for this challenge.

End of code sourcing statement.

 ----------------------
