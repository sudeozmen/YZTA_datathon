import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


#load the data
train = pd.read_csv("train.csv")
test =  pd.read_csv("testFeatures.csv")
test_ids = test["id"]
test = test.drop(columns="id")
test.columns = ['tarih', 'ürün', 'ürün besin değeri', 'ürün kategorisi',
                'ürün üretim yeri', 'market', 'şehir']




#eda
print("\n Train \n")
print(train.info())
print(train.isnull().sum())
print("train duplicate:", train.duplicated())
print(train['ürün kategorisi'].unique())
print(train['ürün üretim yeri'].unique())
print(train['market'].unique())
print(train['şehir'].unique())
sns.histplot(train['ürün fiyatı'])
plt.title("Ürün Fiyat Dağılımı")
plt.show()
print("\n Test Features \n")
print(test.info())
print(test.isnull().sum()) 
print("test duplicate:", test.duplicated())
#there is no null or duplicated value but label encoding shoulde be applied to categorical values and date feature should be rearranged 


#future engineering 
#date feature was split into two columns: year and month 
for df in [train, test]:
    df['tarih'] = pd.to_datetime(df['tarih'])
    df['yıl'] = df['tarih'].dt.year
    df['ay'] = df['tarih'].dt.month
    df.drop('tarih', axis=1, inplace=True)
#label encoding
cat_cols = ['ürün', 'ürün kategorisi', 'ürün üretim yeri', 'market', 'şehir']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le


#random forest regressor
#target variable and the features used for prediction were defined
x = train.drop('ürün fiyatı', axis=1)
y = train['ürün fiyatı']
#model was created and trained
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x,y)
#evaluation
y_pred_train = model.predict(x)
rmse = np.sqrt(mean_squared_error(y,y_pred_train))
print(f"Train RMSE: {rmse:.2f}")
#validation
x_train, x_val, y_train, y_val =train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_val)
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))


#prediction
test_preds = model.predict(test)


#submission file
submission = pd.DataFrame({
    'id' : test_ids,
    'ürün fiyatı' : test_preds
})
submission.to_csv("RFRsub1.csv", index=False)