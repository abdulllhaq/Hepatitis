import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy


# App Title
st.title('Liver Disease Prediction App')

# About Section
st.markdown('''
# Liver Disease Detector

- This app detects if you have a Hepatic (Liver) disease such as Hepatitis, Fibrosis, or Cirrhosis based on Machine Learning!
- App built by Abdul Haq of Team Skillocity.
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.
''')
st.write('---')

# Load Data
try:
    df = pd.read_csv('HepatitisCdata.csv')
except FileNotFoundError:
    st.error("Error: HepatitisCdata.csv not found.  Make sure it's in the same directory.")
    st.stop()

# Preprocessing
df.columns = df.columns.str.strip()  # remove whitespace from column names. Critical fix!
df = df.rename(columns={'Serum Cholinesterase ': 'Serum Cholinesterase'}) # fix a subtle name mismatch


# Data Summary
st.sidebar.header('Patient Data Input')
st.subheader('Dataset Overview')
st.write(df.describe())

# Prepare Data for Model
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Encode target variable
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# User Input Form
def user_input_features():
    age = st.sidebar.slider('Age', int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].mean()))
    albumin = st.sidebar.slider('Albumin', float(X['Albumin'].min()), float(X['Albumin'].max()), float(X['Albumin'].mean()))
    alkaline_phosphate = st.sidebar.slider('Alkaline Phosphate', float(X['Alkaline Phosphate'].min()), float(X['Alkaline Phosphate'].max()), float(X['Alkaline Phosphate'].mean()))
    alanine_aminotransferase = st.sidebar.slider('Alanine Aminotransferase', float(X['Alanine Aminotransferase'].min()), float(X['Alanine Aminotransferase'].max()), float(X['Alanine Aminotransferase'].mean()))
    aspartate_aminotransferase = st.sidebar.slider('Aspartate Aminotransferase', float(X['Aspartate Aminotransferase'].min()), float(X['Aspartate Aminotransferase'].max()), float(X['Aspartate Aminotransferase'].mean()))
    bilirubin = st.sidebar.slider('Bilirubin', float(X['Bilirubin'].min()), float(X['Bilirubin'].max()), float(X['Bilirubin'].mean()))
    serum_cholinesterase = st.sidebar.slider('Serum Cholinesterase', float(X['Serum Cholinesterase'].min()), float(X['Serum Cholinesterase'].max()), float(X['Serum Cholinesterase'].mean()))
    cholesterol = st.sidebar.slider('Cholestrol', float(X['Cholestrol'].min()), float(X['Cholestrol'].max()), float(X['Cholestrol'].mean()))
    creatinine = st.sidebar.slider('Creatinine', float(X['Creatinine'].min()), float(X['Creatinine'].max()), float(X['Creatinine'].mean()))
    gamma_glutamyl_transferase = st.sidebar.slider('Gamma-Glutamyl Transferase', float(X['Gamma-Glutamyl Transferase'].min()), float(X['Gamma-Glutamyl Transferase'].max()), float(X['Gamma-Glutamyl Transferase'].mean()))
    prothrombin = st.sidebar.slider('Prothrombin', float(X['Prothrombin'].min()), float(X['Prothrombin'].max()), float(X['Prothrombin'].mean()))

    data = {
        'Age': age,
        'Albumin': albumin,
        'Alkaline Phosphate': alkaline_phosphate,
        'Alanine Aminotransferase': alanine_aminotransferase,
        'Aspartate Aminotransferase': aspartate_aminotransferase,
        'Bilirubin': bilirubin,
        'Serum Cholinesterase': serum_cholinesterase,
        'Cholestrol': cholesterol,
        'Creatinine': creatinine,
        'Gamma-Glutamyl Transferase': gamma_glutamyl_transferase,
        'Prothrombin': prothrombin
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
predicted_class = label_encoder.inverse_transform(prediction)[0] # Decode prediction
st.subheader('Prediction:')

if predicted_class == 0:
    st.write('Healthy')
elif predicted_class == 1:
    st.write('Hepatitis')
elif predicted_class == 2:
    st.write('Fibrosis')
elif predicted_class == 3:
    st.write('Cirrhosis')

# Model Performance
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'{accuracy * 100:.2f}%')

# Visualization (Example: Feature Importance)
st.subheader('Feature Importance:')
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt) # display plot in streamlit

# Footer
st.write('App built by Abdul Haq.')
st.write('Disclaimer: This is for educational purposes only. Consult a doctor for medical advice.')
