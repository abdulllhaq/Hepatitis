import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
from sklearn import preprocessing

#about 
st.markdown('''
# Liver Disease Detector 
- This app detects if you have Hepatitis, Fibrosis or Cirrhosis based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
- Datset: Cleveland and Hungarian heart disease dataset
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.
  Dataset creators:
- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
''')
st.write('---')

#obtain dataset
df = pd.read_csv(r'HepatitisCdata.csv')

#titles
st.title('Liver Disease Detector')
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())




#training
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)


#user report
def user_report():
  Age = st.sidebar.slider('Age', 15,80, 45)
  Albumin = st.sidebar.slider('Albumin', 12,85, 44)
  Alkaline_Phosphate = st.sidebar.slider('Alkaline Phosphate', 0,420, 70)
  Alanine_Aminotransferase = st.sidebar.slider('Alanine Aminotransferase', 0,330, 30)
  Aspartate_Aminotransferase = st.sidebar.slider('Aspartate Aminotransferase', 10,330, 40)
  Bilirubin = st.sidebar.slider('Bilirubin', 0,260, 12)
  Serum_Cholinesterase = st.sidebar.slider('Serum Cholinesterase', 0,20, 11)
  Cholestrol = st.sidebar.slider('Cholestrol', 0,10, 5)
  Creatinine = st.sidebar.slider('Creatinine', 0,1200, 80)
  Gamma_Glutamyl_Transferase = st.sidebar.slider('Gamma-Glutamyl Transferase', 0,700, 42)
  Prothrombin = st.sidebar.slider('Prothrombin', 0,100, 72)

 
  
  
  
  user_report_data = {
      'Age':Age,
      'Albumin':Albumin,
      'Alkaline Phosphate':Alkaline_Phosphate,
      'Alanine Aminotransferase':Alanine_Aminotransferase,
      'Aspartate Aminotransferase':Aspartate_Aminotransferase,
      'Bilirubin':Bilirubin,
      'Serum Cholinesterase':Serum_Cholinesterase,
      'Cholestrol':Cholestrol,
      'Creatinine':Creatinine,
      'Gamma-Glutamyl Transferase':Gamma_Glutamyl_Transferase,
      'Prothrombin':Prothrombin,  
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)





rf  = RandomForestClassifier()
rf.fit(x_train, training_scores_encoded)
user_result = rf.predict(user_data)




st.title('Graphical Patient Report')




if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


#rbp
st.header('Albumin Value Graph (Yours vs Others)')
fig_Albumin = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Albumin', data = df, hue = 'Outcome' , palette='rocket')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Albumin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(10,90,5))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Albumin)


#chol, 2013 damn it
st.header('Alkaline Phosphate Value Graph (Yours vs Others)')
fig_Alkaline_Phosphate = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Alkaline Phosphate', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Alkaline Phosphate'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,440,20))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Alkaline_Phosphate)

#Hmax
st.header('Alanine Aminotransferase Value Graph (Yours vs Others)')
fig_Alanine_Aminotransferase = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Alanine Aminotransferase', data = df, hue = 'Outcome', palette='mako')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['Alanine Aminotransferase'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,340,20))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Alanine_Aminotransferase)


#STDIE
st.header('Aspartate Aminotransferase Value Graph (Yours vs Others)')
fig_Aspartate_Aminotransferase = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Aspartate Aminotransferase', data = df, hue = 'Outcome', palette='flare')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['Aspartate Aminotransferase'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,340,20))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Aspartate_Aminotransferase)


#FCV
st.header('Bilirubin Value Graph (Yours vs Others)')
fig_Bilirubin = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Bilirubin', data = df, hue = 'Outcome', palette='crest')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Bilirubin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,300,20))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Bilirubin)

st.header('Serum Cholinesterase	 Value Graph (Yours vs Others)')
fig_Serum_Cholinesterase = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Serum Cholinesterase	', data = df, hue = 'Outcome', palette='magma')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Serum Cholinesterase'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,20,1))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Serum_Cholinesterase)

st.header('Cholestrol Value Graph (Yours vs Others)')
fig_Cholestrol = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Cholestrol', data = df, hue = 'Outcome', palette='viridis')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Cholestrol'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,10,0.5))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Cholestrol)

st.header('Creatinine Value Graph (Yours vs Others)')
fig_Creatinine = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Creatinine', data = df, hue = 'Outcome', palette='coolwarm')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Creatinine'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,1200,50))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Creatinine)

st.header('Gamma-Glutamyl Transferase Value Graph (Yours vs Others)')
fig_Gamma_Glutamyl_Transferase = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Gamma-Glutamyl Transferase', data = df, hue = 'Outcome', palette='icefire')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Gamma-Glutamyl Transferase'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,700,25))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Gamma_Glutamyl_Transferase)

st.header('Prothrombin Value Graph (Yours vs Others)')
fig_Prothrombin = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Prothrombin', data = df, hue = 'Outcome', palette='Spectral')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Prothrombin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,100,5))
plt.title('0 - Healthy, 1 - Hepatitis, 2 - Fibrosis, 3 - Cirrhosis')
st.pyplot(fig_Prothrombin)




#Final Report
st.subheader('Your Report: ')
st.title(len(user_result[]))
#st.title(user_result[2])

#if user_result[0]==0:
  #output = 'Congratulations, you do not have any liver diseases.'
#elif user_result[0]==1:
  #output = "Unfortunately, you do have Hepatitis."
#elif user_result[0]==2:
  #output = "Unfortunately, you do have Fibrosis."
#else:
  #output = 'Unfortunately, you do have Cirrosis.'
#st.title(output)




#Most important for users
st.subheader('Lets raise awareness for cardiovascular health and increase awareness about cardiovascular diseases.')
st.write("World Heart Day: 29 September")

st.sidebar.subheader("""An article about this app: https://proskillocity.blogspot.com/2021/05/heart-disease-detector-web-app.html""")
st.write("Dataset License: Creative Commons Attribution 4.0 International (CC BY 4.0)")

st.write("Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have any cardiovascular diseases or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")

image = Image.open('killocity (3).png')

st.image(image, use_column_width=True)
