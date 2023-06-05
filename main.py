import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import streamlit as st
from sklearn.pipeline import Pipeline

url = 'https://github.com/filopacio/Cardio-NN-Classification-/blob/main/cardio.csv?raw=true'
data = pd.read_csv(url, sep = ';')
data = data.iloc[:,:-2]
data = data[['totchol', 'sysbp', 'glucose', 'age', 'bmi', 'cursmoke', 'diabetes', 'sex', 'event']]
X = data.drop('event', axis=1)
y = data['event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def encode_labels(column):
    if len(column.unique()) == 2:
       encoded_column = column.map({'Yes': 1, 'No': 0})    
       return encoded_column
    else:
      return column



X_train_encoded = X_train.apply(encode_labels)
X_test_encoded = X_test.apply(encode_labels)
y_train_encoded = encode_labels(y_train)
y_test_encoded = encode_labels(y_test)

lazy_model = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_model.fit(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded)

models_dict = lazy_model.provide_models(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded)
best_model = models_dict[models.index[0]]

pipeline = Pipeline(best_model.steps)

# Step 5: Deploy the Streamlit app
st.title("Medical Cure Prediction")
sysbp = st.slider("Systolic Blood Pressure", 80, 200)
glucose = st.slider("Glucose Level", 50, 300)
totchol = st.slider("Total Cholesterol", 100, 400)
age = st.slider("Age", 20, 90)
bmi = st.slider("BMI", 10, 50)
cursmoke = st.checkbox("Smoker")
diabetes = st.checkbox("Diabetes")
sex = st.selectbox("Sex", ("Male", "Female"))
sex_binary = 1 if sex == "Female" else 0
features = [totchol, age, sysbp, cursmoke, bmi, diabetes, glucose, sex_binary]
X_pred = pd.DataFrame({'totchol': [totchol],
                     'sysbp': [sysbp],
                     'glucose': [glucose],
                     'age': [age],
                     'bmi': [bmi],
                     'cursmoke': [cursmoke],
                     'diabetes': [diabetes],
                     'sex': [sex_binary]})
prediction = pipeline.predict(X_pred)

if prediction ==1:
    st.title(":red[You may need a medical treatment]")
else:
    st.title(":green[No medical treatment needed]")
