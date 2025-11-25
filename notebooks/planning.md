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




### Some papers
- https://mjs.um.edu.my/index.php/MJS/article/view/28583?utm_source=chatgpt.com
- https://www.mdpi.com/2227-9091/11/12/213?utm_source=chatgpt.com
- https://econpapers.repec.org/article/vrsaicuec/v_3a62_3ay_3a2015_3ai_3a2_3ap_3a151-168_3an_3a2.htm?utm_source=chatgpt.com


 sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator, TransformerMixin):
def __init__(self, n_clusters=10, gamma=1.0, random_state=None):



def column_ratio(X):
return X[:, [0]] / X[:, [1]]
def ratio_name(function_transformer, feature_names_in):
return ["ratio"] # feature names out
def ratio_pipeline():
return make_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(column_ratio, feature_names_out=ratio_name),
StandardScaler())
log_pipeline = make_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(np.log, feature_names_out="one-to-one"),
StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
StandardScaler())
preprocessing = ColumnTransformer([
("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
("people_per_house", ratio_pipeline(), ["population", "households"]),
("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
"households", "median_income"]),
("geo", cluster_simil, ["latitude", "longitude"]),
("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
remainder=default_num_pipeline) # one column remaining: housing_median_age

something similar for pipeline