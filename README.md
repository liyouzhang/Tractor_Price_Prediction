# Tractor_Price_Prediction
Case Study at Galvanize Data Science Immersive Program.

2018 June 25th

## Infomation
We want to leverage the real world data to develop a predictive model, to help tractor buyers and sellers to have an estimated price for their reference.

Team Member: Josh, Liyou, Hamilton, Andrew


## Findings

* The best predictors for the auction price of a machine are Product Group, Product Size, YearMade, MachineHoursCurrentMeter, Age, auctioneer.


## Methodology

* The dataset contains 53 features of tractors and have more than 400,000 rows. Most of the features have high percentage of missing values.

* Feature engineering includes filling NaNs with medians, transform datatype, calculate age of the tractor and create dummies for categorical variables.

* Used linear regression model and the measurement of performance is Root Mean Squared Log Error (RMSLE).
