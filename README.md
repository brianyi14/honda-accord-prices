# Honda Accord Prices
Brian Yi

## Introduction

This project will be split into two parts:

**Part 1:**

We will be focusing on modeling and statistical analysis using the following models:

    - Single Linear Regression
    - Multiple Linear Regression
    - Polynomial Regression (one variable)
    - Polynomial Regression (two variables)

The dataset we will be using is the UsedCarLot dataset has the following five variables: `age`, `price`, `mileage`, `model`, `make`, and `year`. Our target variable to predict is `price`, and the predictors we will focus on are `age` and `mileage`. We don't consider `model` and `make` since we will only be analyzing the price of Honda Accords. We will use residual analysis visualizations and various hypothesis tests to evaluate individual models. The nested F-test will be our main test used to compare different models and help us to determine the best model for predicting Honda Accord prices.

**Part 2:**

We will be focusing on one-way ANOVA and two-way ANOVA tests for this part to investigate the variance withing roup means within the `UsedCar` dataframe we will create. We will add two new variables, `type` and `country`, in order to conduct these tests. `type` will be either sedan or SUV depending on the car. `country` represents a car's country of manufacture, which will either be Japan, Germany, or the US.

For our one-way ANOVA test, the independent variable we are interested in is the `model` variable with respect to `price`. In other words, we want to determine whether different `model` cars have different `prices`.

For our two-way ANOVA test, the independent variables we are interested are `type` and `country` with respect to `price`. This time, we want to see whether a different `type` of car with different `country` origins will have different `prices`.
