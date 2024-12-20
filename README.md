# House_Prices_Advanced_Regression_Techniques
Predict the sales price for each house


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

# Import Data


```python
# Load the training dataset
df_train = pd.read_csv(r"C:\Users\ACER\Downloads\house-prices-advanced-regression-techniques\train.csv", index_col='Id')

# Drop unnecessary columns
df_train.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)

```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>...</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>...</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 74 columns</p>
</div>




```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1460 entries, 1 to 1460
    Data columns (total 74 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1460 non-null   int64  
     1   MSZoning       1460 non-null   object 
     2   LotFrontage    1201 non-null   float64
     3   LotArea        1460 non-null   int64  
     4   Street         1460 non-null   object 
     5   LotShape       1460 non-null   object 
     6   LandContour    1460 non-null   object 
     7   Utilities      1460 non-null   object 
     8   LotConfig      1460 non-null   object 
     9   LandSlope      1460 non-null   object 
     10  Neighborhood   1460 non-null   object 
     11  Condition1     1460 non-null   object 
     12  Condition2     1460 non-null   object 
     13  BldgType       1460 non-null   object 
     14  HouseStyle     1460 non-null   object 
     15  OverallQual    1460 non-null   int64  
     16  OverallCond    1460 non-null   int64  
     17  YearBuilt      1460 non-null   int64  
     18  YearRemodAdd   1460 non-null   int64  
     19  RoofStyle      1460 non-null   object 
     20  RoofMatl       1460 non-null   object 
     21  Exterior1st    1460 non-null   object 
     22  Exterior2nd    1460 non-null   object 
     23  MasVnrArea     1452 non-null   float64
     24  ExterQual      1460 non-null   object 
     25  ExterCond      1460 non-null   object 
     26  Foundation     1460 non-null   object 
     27  BsmtQual       1423 non-null   object 
     28  BsmtCond       1423 non-null   object 
     29  BsmtExposure   1422 non-null   object 
     30  BsmtFinType1   1423 non-null   object 
     31  BsmtFinSF1     1460 non-null   int64  
     32  BsmtFinType2   1422 non-null   object 
     33  BsmtFinSF2     1460 non-null   int64  
     34  BsmtUnfSF      1460 non-null   int64  
     35  TotalBsmtSF    1460 non-null   int64  
     36  Heating        1460 non-null   object 
     37  HeatingQC      1460 non-null   object 
     38  CentralAir     1460 non-null   object 
     39  Electrical     1459 non-null   object 
     40  1stFlrSF       1460 non-null   int64  
     41  2ndFlrSF       1460 non-null   int64  
     42  LowQualFinSF   1460 non-null   int64  
     43  GrLivArea      1460 non-null   int64  
     44  BsmtFullBath   1460 non-null   int64  
     45  BsmtHalfBath   1460 non-null   int64  
     46  FullBath       1460 non-null   int64  
     47  HalfBath       1460 non-null   int64  
     48  BedroomAbvGr   1460 non-null   int64  
     49  KitchenAbvGr   1460 non-null   int64  
     50  KitchenQual    1460 non-null   object 
     51  TotRmsAbvGrd   1460 non-null   int64  
     52  Functional     1460 non-null   object 
     53  Fireplaces     1460 non-null   int64  
     54  GarageType     1379 non-null   object 
     55  GarageYrBlt    1379 non-null   float64
     56  GarageFinish   1379 non-null   object 
     57  GarageCars     1460 non-null   int64  
     58  GarageArea     1460 non-null   int64  
     59  GarageQual     1379 non-null   object 
     60  GarageCond     1379 non-null   object 
     61  PavedDrive     1460 non-null   object 
     62  WoodDeckSF     1460 non-null   int64  
     63  OpenPorchSF    1460 non-null   int64  
     64  EnclosedPorch  1460 non-null   int64  
     65  3SsnPorch      1460 non-null   int64  
     66  ScreenPorch    1460 non-null   int64  
     67  PoolArea       1460 non-null   int64  
     68  MiscVal        1460 non-null   int64  
     69  MoSold         1460 non-null   int64  
     70  YrSold         1460 non-null   int64  
     71  SaleType       1460 non-null   object 
     72  SaleCondition  1460 non-null   object 
     73  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(34), object(37)
    memory usage: 855.5+ KB
    

# Splitting Data and Preprocessing


```python
# Split into features and target
X = df_train.drop(columns='SalePrice')
y = df_train['SalePrice']
```


```python
# Define numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
```


```python
# Define preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer([
    ('numeric', numerical_pipeline, numerical_features),
    ('categoric', categorical_pipeline, categorical_features)
])
```

# Modelling, Tuning, and Evaluation


```python
# Define the XGBRegressor model pipeline
pipeline_xgb = Pipeline([
    ('prep', preprocessor),
    ('algo_xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])
```


```python
# Define the parameter grid for XGBRegressor
param_grid_xgb = {
    'algo_xgb__n_estimators': [100, 200],
    'algo_xgb__learning_rate': [0.05, 0.1],
    'algo_xgb__max_depth': [3, 5],
    'algo_xgb__min_child_weight': [1, 3],
    'algo_xgb__gamma': [0, 0.1],
    'algo_xgb__alpha': [0, 0.1, 1, 10],  # L1 Regularization
    'algo_xgb__lambda': [1, 2, 5, 10]   # L2 Regularization
}

# Define custom scoring (RMSE)
scoring = 'neg_root_mean_squared_error'

# Grid search for XGBRegressor
grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid=param_grid_xgb, cv=3, scoring=scoring, n_jobs=-1, verbose=1)

# Fit the grid search object on training data
grid_search_xgb.fit(X, y)

# Print the best parameters and best RMSE score for XGBRegressor
print("Best parameters for XGBRegressor:", grid_search_xgb.best_params_)
print("Best RMSE score for XGBRegressor:", -grid_search_xgb.best_score_)

# Print R-squared score for XGBRegressor with best parameters
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_train = best_xgb_model.predict(X)
r2_xgb_train = r2_score(y, y_pred_train)
print("R^2 score for XGBRegressor on training data:", r2_xgb_train)
```

    Fitting 3 folds for each of 512 candidates, totalling 1536 fits
    Best parameters for XGBRegressor: {'algo_xgb__alpha': 0.1, 'algo_xgb__gamma': 0, 'algo_xgb__lambda': 10, 'algo_xgb__learning_rate': 0.1, 'algo_xgb__max_depth': 5, 'algo_xgb__min_child_weight': 1, 'algo_xgb__n_estimators': 200}
    Best RMSE score for XGBRegressor: 26785.23880409128
    R^2 score for XGBRegressor on training data: 0.9894520044326782
    


```python
# Display actual vs predicted SalePrice for training data
results_train = pd.DataFrame({'Actual_SalePrice': y, 'Predicted_SalePrice': y_pred_train})
print("\nActual vs Predicted SalePrice for training data:")
print(results_train.head())
```

    
    Actual vs Predicted SalePrice for training data:
        Actual_SalePrice  Predicted_SalePrice
    Id                                       
    1             208500        200526.234375
    2             181500        175038.375000
    3             223500        213689.109375
    4             140000        151076.687500
    5             250000        266002.906250
    


```python
import matplotlib.pyplot as plt

# Plot actual vs predicted for training set
plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred_train, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='red')
plt.title("XGBRegressor: Actual vs Predicted SalePrice on Training Data")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.grid(True)
plt.show()
```


    
<img src = 'https://github.com/anggapradanaa/House_Prices_Advanced_Regression_Techniques/blob/main/actual%20vs%20predicted%20for%20training%20set.png'>
    


# Apply to New Dataset and Predict


```python
# Load the test dataset
df_test = pd.read_csv(r"C:\Users\ACER\Downloads\house-prices-advanced-regression-techniques\test.csv", index_col='Id')  # Adjust path as per your test dataset location
```


```python
# Preprocess the test dataset
X_test_new = grid_search_xgb.best_estimator_.named_steps['prep'].transform(df_test)

# Make predictions on the test dataset
predictions = grid_search_xgb.best_estimator_.named_steps['algo_xgb'].predict(X_test_new)

```


```python
# Create a DataFrame with 'Id' and predicted 'SalePrice'
results_df = pd.DataFrame({
    'Id': df_test.index,
    'SalePrice': predictions
})

# Display the first few rows of the results
print("\nSample of 'Id' and 'SalePrice' columns from test dataset:")
print(results_df.head())
```

    
    Sample of 'Id' and 'SalePrice' columns from test dataset:
         Id      SalePrice
    0  1461  125369.429688
    1  1462  155000.562500
    2  1463  182293.890625
    3  1464  192370.625000
    4  1465  189375.515625
    


```python
# Save the predictions to a CSV file
results_df.to_csv(r"C:\Users\ACER\Downloads\House Prices - Advanced Regression Techniques.csv", index=False)

print("Predictions saved to 'predictions.csv'.")

```

    Predictions saved to 'predictions.csv'.
    
