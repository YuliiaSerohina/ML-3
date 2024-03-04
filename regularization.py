import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib

# 2 Дослідити вплив регуляризації на точність моделі лінійної регресії

dataset = pd.read_csv('kc_house_data.csv')
print(dataset.head())

y = np.asarray(dataset['price'].values.tolist())
y = y.reshape(len(y), 1)
x = np.array(dataset.drop(['price', 'date', 'id'], axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Elastic Net': ElasticNet()
}

for name, model in models.items():
    try:
        model.fit(x_train_scaler, y_train)
        y_pred = model.predict(x_test_scaler)
        mse = mean_squared_error(y_test, y_pred)
        score = model.score(x_test_scaler, y_test)
        print(f"{name}: MSE = {mse}, score = {score}")
    except Exception as e:
        print(f"Error occurred while fitting {name}: {e}")


# Лінійна регресія без регуляризації показує велику помилку. Моделі з регуляризацією Lasso, Ridge, Elastic Net
#показують приблизно однакову помилку, яка значно менше за LinearRegression. Це показує, що використання регуляризації
#допомогло зменшити перенавчання(як допущення) моделей та повисити узагальнуючу властивість моделей

#2.2 Робота з CV датасетом

x_train2, x_other, y_train2, y_other = train_test_split(x, y, test_size=0.4, random_state=42)
x_eval, x_test2, y_eval, y_test2 = train_test_split(x_other, y_other, test_size=0.5, random_state=42)

x_train_scaler = scaler.fit_transform(x_train2)
x_eval_scaler = scaler.transform(x_eval)
x_test_scaler = scaler.transform(x_test2)

eval_rmse_errors = []
d_range = np.arange(1, 5)

for d in d_range:
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    x_poly_train = poly_converter.fit_transform(x_train_scaler)
    x_poly_eval = poly_converter.fit_transform(x_eval_scaler)
    model = Ridge(alpha=1)
    model.fit(x_poly_train, y_train2)
    y_pred = model.predict(x_poly_eval)
    RMSE = np.sqrt(mean_squared_error(y_eval, y_pred))
    eval_rmse_errors.append(RMSE)
print(eval_rmse_errors)
optimal_d = d_range[np.argmin(np.array(eval_rmse_errors))]
print(optimal_d)

poly_converter_2 = PolynomialFeatures(degree=optimal_d, include_bias=False)
poly_features_train_2 = poly_converter_2.fit_transform(x_train_scaler)
poly_features_test_2 = poly_converter_2.fit_transform(x_test_scaler)
final_model = Ridge(alpha=1)
final_model.fit(poly_features_train_2, y_train2)
final_predict = final_model.predict(poly_features_test_2)
RMSE = np.sqrt(mean_squared_error(y_test2, final_predict))
print('RMSE: ', RMSE)
joblib.dump(final_model, "final_model.joblib")






