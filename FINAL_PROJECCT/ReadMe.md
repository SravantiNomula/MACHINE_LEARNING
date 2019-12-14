For the final project of the course we had to pick a project from a given list and compete againt each other on the leader board
The project we selected was : House price prediction
Kaggle Competition: https://www.kaggle.com/c/house-price-predictioniiitb/leaderboard

Team Members: Sravanti Nomula (MT2018524) Sarvesh Nandkar(MT2018519)

In the project we have mainly focused on pre-processing  as the data had many null entries and Model building. The missing values have been dealt differently depending upon the feature they belong to. We have dropped features like ‘PoolQc’ which had nearly all the entries missing. We have imputed the missing values using the probability in categorical features like ‘MiscFeature’. We have used groupby feature ‘Neighbourhood’ to impute. We have simply replaced the missing values with ‘none’ or ‘0’ in other features. We have used several models for the project from basic ones like linear regression to XGBoost. We have also made models by stacking and ensembling few stand alone models gave lesser error than the ensembled ones but as we wanted the model to be more generalized and less overfitting we went for the ensembled model. 
