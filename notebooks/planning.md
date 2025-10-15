# Project planning

## Look at the big picture

### 1. Define the objective in business terms.

We have to predict the expected number of insurance claims made by customers over a year based on vehicle and driver information. This helps the insurance company to assess risk more accurately, price policies fairly and reduce losses. Measuring claims risk is a hard job, given the data we have to come up with good measures.

### 2. How will the solution be used?

Insurance company can set the prices accurately, and charge higher for high-risk drivers, and lower it for low-risk ones. Identify potential fraud. Allocate resources effectively. It can help the data science team to track and improve predictive performance over time.

### 3. Frame of the problem

The type of machine learning problem is given: supervised learning, as we have labels(ClaimNb) and its a regression task as long as we keep the claims numeric, and not as categories(low/medium/high). Batch learning as we have historical data, no continious data incoming, we train and evaluate models before predicting future policies. We could theoretically convert `ClaimNb` into discrete categories (e.g., low/medium/high risk) and use classification models. However, this would be a deviation from the guidelines and would require justification for the chosen bins. For consistency with the course instructions and to align with the business objective of estimating expected claims, we treat the target as a **continuous numeric variable**. We need further research to determine the exact target variable.
`ClaimNb` should be divided by `Exposure` to create a normalized target, `Claims per year`.

### 4. Metric

The primary objective is to predict the expected number of claims per policyholder. If we stay with numeric labels, we have multiple metrics that could be used:

1. **Mean Absolute Error (MAE)** – measures average deviation between predicted and actual claim counts.
2. **Root Mean Squared Error (RMSE)** – penalizes large deviations more heavily, useful for identifying models that underestimate high-risk policies.
3. **Poisson Deviance (optional)** – statistically appropriate for count data like claims.
   RMSE seems like a good choice as it penalizes large mistakes.

### 5. Assumptions
