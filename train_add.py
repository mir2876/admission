import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load data
admission = pd.read_csv("D:/Bootcamp ML & AI/admission_app/admission_data.csv")
admission.info()
# check duplicated data overall
admission.duplicated().sum()
admission[admission.duplicated(keep=False)]
#Ada 5 data yang duplicated
#drop data duplicated
admission = admission.drop_duplicates()
# check missing values in each column
admission.isna().sum()
# percentage of missing values in each column
admission.isna().mean()
# drop all rows with missing values
admission = admission.dropna()
#imputasi
#admission['gre_score'].fillna(admission['gre_score'].mean(), inplace=True)
#admission['toefl_score'].fillna(admission['toefl_score'].mean(), inplace=True)
#admission['motiv_letter_strength'].fillna(admission['motiv_letter_strength'].mean(), inplace=True)
#admission['recommendation_strength'].fillna(admission['recommendation_strength'].mean(), inplace=True)
#admission['gpa'].fillna(admission['gpa'].mean(), inplace=True)
#admission['research_exp'].fillna(admission['research_exp'].mode()[0], inplace=True)
# draw histogram for each numerical column
admission.hist(figsize=(15, 10))
plt.show()
#semua sudah berdistribusi normal
#Cek Outlier
# draw boxplot for each numeric column
plt.figure(figsize=(12,6))

# plotting
features = ['gre_score','toefl_score','motiv_letter_strength','recommendation_strength','gpa','research_exp']
for i in range(0, len(features)):
    plt.subplot(1, len(features), i+1)
    sns.boxplot(y=admission[features[i]], color='red')
    plt.tight_layout()
# drop rows that have outliers of recommendation_strength
# Using IQR method
Q1 = admission['recommendation_strength'].quantile(0.25)
Q3 = admission['recommendation_strength'].quantile(0.75)
IQR = Q3 - Q1

admission = admission[~((admission['recommendation_strength'] < (Q1 - 1.5 * IQR)) | (admission['recommendation_strength'] > (Q3 + 1.5 * IQR)))]
# value counts of categorical columns in admission
features = ['research_exp','admit_status','univ_tier']

for feature in features:
    print("***"*10)
    print(f'Value Counts of {feature}')
    print(admission[feature].value_counts())
    print('\n')
#Feature encoding
# label encode research_exp
research_exp_map = {
    'yes': 1,
    'no': 0
}

admission['research_exp'] = admission['research_exp'].map(research_exp_map)
# label encode admit_status
admit_status_map = {
    'yes': 1,
    'no': 0
}
admission['admit_status'] = admission['admit_status'].map(admit_status_map)
# label encode univ_tier
univ_tier_map = {
    'high': 1,
    'low': 0
}
admission['univ_tier'] = admission['univ_tier'].map(univ_tier_map)
# min-max scaling all column
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for col in admission.columns:
    admission[col] = scaler.fit_transform(admission[[col]])
admission.describe()

# split train test
from sklearn.model_selection import train_test_split

feature = admission.drop(columns='admit_status')
target = admission[['admit_status']]

feature_admit_train, feature_admit_test, target_admit_train, target_admit_test = train_test_split(feature, target, test_size=0.20, random_state=42)
#save data to csv
admission.to_csv('D:/Bootcamp ML & AI/admission_app/admission_clean.csv', index=False)
#Multicolinearity handling
# calculate vif score for each column in feature_admit_train
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

X = add_constant(feature_admit_train)

vif_add = (pd.DataFrame(
            [vif(X.values, i) for i in range(len(X.columns))]
            ,index=X.columns)
            .reset_index())

vif_add.columns = ['feature','vif_score']
vif_add = vif_add.loc[vif_add.feature!='const']
vif_add
# heatmap correlation
admit_train = pd.concat([feature_admit_train, target_admit_train], axis=1)
corr = admit_train.corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()
#drop gre dan toefl
# drop fitur yang redundan
feature_admit_train = feature_admit_train.drop(columns=['gre_score','toefl_score'])
feature_admit_test = feature_admit_test.drop(columns=['gre_score','toefl_score'])
#regresi lasso
from sklearn.linear_model import Lasso

# define the model
lasso_reg = Lasso(alpha=10,
                  random_state=42)

# train
lasso_reg.fit(feature_admit_train, target_admit_train)
#memunculkan hasil estimasi parameter
# retrieve the coefficients
# show as a nice dataframe

data = feature_admit_train
model = lasso_reg

coef_add = pd.DataFrame({
    'feature':['intercept'] + data.columns.tolist(),
    'coefficient':[model.intercept_] + list(model.coef_)
})

coef_add

#Hyperparameter tuning dengan lasso CV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50),  # Coba alpha dari 0.0001 sampai 10
    cv=5,                           # 5-fold cross validation
    random_state=42
)
lasso_cv.fit(feature_admit_train, target_admit_train)
#Hasil alpha optimal dan koefisien model
print(f"Alpha optimal: {lasso_cv.alpha_:.6f}\n")

coef = pd.DataFrame({
    'feature':['intercept'] + feature_admit_train.columns.tolist(),
    'coefficient':[lasso_cv.intercept_] + list(lasso_cv.coef_)
})

coef

lasso_best=lasso_cv

#Model Evaluation
# prepare prediction result on train data
target_predict_train = lasso_best.predict(feature_admit_train)
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

print('RMSE for training data is {}'.format(root_mean_squared_error(target_admit_train, target_predict_train)))
print('MAE for training data is {}'.format(mean_absolute_error(target_admit_train, target_predict_train)))
print('MAPE for training data is {}'.format(mean_absolute_percentage_error(target_admit_train, target_predict_train)))

#Testing
target_predict_test = lasso_best.predict(feature_admit_test)
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

print('RMSE for testing data is {}'.format(root_mean_squared_error(target_admit_test, target_predict_test)))
print('MAE for testing data is {}'.format(mean_absolute_error(target_admit_test, target_predict_test)))
print('MAPE for testing data is {}'.format(mean_absolute_percentage_error(target_admit_test, target_predict_test)))

#Simpan hasil train
import pickle
with open("D:/Bootcamp ML & AI/admission_app/hasil.pkl", "wb") as file:
    pickle.dump(lasso_best, file)