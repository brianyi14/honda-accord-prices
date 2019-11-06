# Honda Accord Prices

Brian Yi

## Introduction

**Purpose:** When looking to purchase a used car, I always find it difficult to gauge prices since used cars vary in mileage and age. Furthermore, I don't know if there is a significant difference in pricing between German, Japanese, or American cars of varying types (sedan or SUV). It is especially difficult to tell which car model is most worth its price. Even though these questions have more definitive answers when purchasing a new car, it can be quite different for used cars since certain cars may depreciate faster than others. In this project, we are building a model that can assist in evaluating the price of used cars based on all these features.

**Method of Approach:** We will be using the UsedCarLot dataset that has the following five variables: `age`, `price`, `mileage`, `model`, `make`, and `year`. This project will be split into two parts in predicting our response variable, `price`:

**Part 1:** Since I want to purchase a Honda Accord, we will determine what prices I should be expecting based on an Accord's `age` and `mileage`. We build a few different linear and polynomial regression models with these two variables to predict `price`. We do some hypothesis testing to briefly evaluate these models before using the nested F-test to determine the best model. Next, we conduct some residual analysis for our best model to check for constant variance, normality, and zero mean. Finally, we take our model out for a spin and predict the prices for a Honda Accord that I would be looking to buy.

**Part 2:** For the second part of this project, we want to see if the average prices of cars are different between various car models. We use a one-way ANOVA test, with `model` as the predictor and `price` as the response variable, to detect if any car model has a different mean price from the others.

We also want to determine whether cars with a different country of manufacture and of a different type (sedan or SUV) have a different mean price. Therefore, we add two new predictors, `type` and `country`, in order to conduct this analysis. We use a two-way ANOVA test, with the independent variables being `type` and `country`, to predict `price`.

**Results:** The model we found (multiple linear regression with `age` and `mileage` as predictors) for predicting Honda Accord prices did a good job in fitting our dataset based on the metrics we evaluated it with. Our analysis of whether car prices differed based on model, type (sedan or SUV), and country of manufacture revealed that German cars were the most expensive. We do note that this result is heavily influenced by our limited dataset.
