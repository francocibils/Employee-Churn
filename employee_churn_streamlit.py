import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load model
rf_clf = pickle.load(open('Employee-Churn/rf_clf.pkl', 'rb'))

# Naming
areas = ['Product management', 'Technical', 'Sales', 'Management', 'Accounting', 'R+D', 'Marketing', 'HHRR', 'Support', 'IT']
areas_real = ['gestiÃ³n de productos', 'tecnica', 'ventas', 'direcciÃ³n', 'contabilidad', 'ImÃ¡sD', 'marketing', 'RRHH', 'soporte', 'TI']

salarios = ['Low', 'Medium', 'High']
salarios_real = ['bajo', 'medio', 'alto']

areas_dict = dict(zip(areas, areas_real))
salarios_dict = dict(zip(salarios, salarios_real))
yes_no_dict = {'Yes': 1., 'No': 0.}

# App
st.title('Employee Churn - Prediction App')
st.markdown('This is a simple employee churn prediction app. The aim is to provide a tool that allows a HR user to assess the likelihood that an employee will leave the company. For that purpose the HR user should input the characteristics of a certain employee and the app will return the probability of churn for that particular employee.')

st.subheader('Fill with characteristics of the employee')

satisfaction = st.slider('Satisfaction level', min_value = 0., max_value = 1., step = 0.01)
review = st.slider('Last performance review score', min_value = 0., max_value = 1., step = 0.01)
projects = st.number_input('Amount of projects that the employee is involved in', min_value = 0)
hours_worked = st.number_input('Average of monthly hours worked', min_value = 0)
years_company = st.number_input('Years at the company', min_value = 0)
accident = st.radio('Has the employee had a work accident?', ['Yes', 'No'])
promotion = st.radio('Has the employee been promoted in the last 5 years?', ['Yes', 'No'])
area = st.selectbox('In what department does the employee work?', areas)
salary = st.selectbox("What is the employee's salary level?", salarios)

# Prediction
pred_dict = {
    'nivel_de_satisfaccion': satisfaction,
    'ultima_evaluacion': review,
    'cantidad_proyectos': projects,
    'promedio_horas_mensuales_trabajadas': hours_worked,
    'años_en_la_empresa': years_company,
    'tuvo_un_accidente_laboral': accident,
    'promociones_ultimos_5_anios': promotion,
    'area': area,
    'salario': salary
}

if st.button('Predict'):

    pred_df = pd.DataFrame.from_dict(pred_dict, orient = 'index').T
    pred_df = pred_df.replace(areas_dict)
    pred_df = pred_df.replace(salarios_dict)
    pred_df = pred_df.replace(yes_no_dict)

    pred_df = pd.get_dummies(pred_df, columns = ['area', 'salario'], drop_first = False)

    # Add missing columns
    for i in areas_real:
        if 'area_' + i not in pred_df.columns:
            pred_df['area_' + i] = 0.
            
    for i in salarios_real:
        if 'salario_' + i not in pred_df.columns:
            pred_df['salario_' + i] = 0.

    # Reorder columns (hard-coded, IMPROVE)
    pred_df = pred_df[['nivel_de_satisfaccion', 'ultima_evaluacion', 'cantidad_proyectos',
       'promedio_horas_mensuales_trabajadas', 'años_en_la_empresa',
       'tuvo_un_accidente_laboral', 'promociones_ultimos_5_anios',
       'area_ImÃ¡sD', 'area_RRHH', 'area_TI', 'area_contabilidad',
       'area_direcciÃ³n', 'area_gestiÃ³n de productos', 'area_marketing',
       'area_soporte', 'area_tecnica', 'area_ventas', 'salario_alto',
       'salario_bajo', 'salario_medio']]

    prob_churn = rf_clf.predict_proba(pred_df)[:, 1][0] * 100

    st.markdown(f'The probability that the employee with the specified characteristics will leave the company is **{round(prob_churn, 2)}%**.')
