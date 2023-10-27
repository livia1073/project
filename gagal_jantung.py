import pickle
import streamlit as st

model = pickle.load(open('gagal_jantung.sav', 'rb'))

st.title('PERHITUNGAN RESIKO TERKENA GAGAL JANTUNG')

age = st.text_input('Usia Anda')
anaemia = st.selectbox('Apakah anda memiliki anemia', ['Ya', 'Tidak'])

if anaemia == 'Ya':
    anaemia = 1
else:
    anaemia = 0
creatinine_phosphokinase = st.text_input('Level dari CPK enzyme dalam darah (mcg/L)')
diabetes = st.selectbox('Apakah anda memiliki diabetes', ['Ya', 'Tidak'])

if diabetes == 'Ya':
    diabetes = 1
else:
    diabetes = 0
ejection_fraction = st.text_input('Persentase darah yang dikeluarkan jantung setiap detaknya')
high_blood_pressure = st.selectbox('Apakah anda memiliki darah tinggi', ['Ya', 'Tidak'])

if high_blood_pressure == 'Ya':
    high_blood_pressure = 1
else:
    high_blood_pressure = 0
platelets = st.text_input('Jumlah kandungan platelets dalah darah (kiloplatelets/mL)')
serum_creatinine = st.text_input('Jumlah kandungan serum creatinine dalam darah (mg/dL)')
serum_sodium = st.text_input('Jumlah kandungan serum sodium dalam darah (mEq/L)')
sex = st.selectbox('Jenis Kelamin anda ?', ['Laki laki', 'Wanita'])

if sex == 'Laki laki':
    sex = 1
else:
    sex = 2
smoking = st.selectbox('Apakah anda perokok', ['Ya', 'Tidak'])

if smoking == 'Ya':
    smoking = 1
else:
    smoking = 0
time = st.text_input('Hari hingga jadwal kontrol selanjutnya')

resiko = ''

if st.button('Tingkat Resiko'):
    tingkat_resiko = model.predict([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
    
    if(tingkat_resiko[0] == 0):
        resiko = 'Anda tidak beresiko terkena gagal jantung'
    else :
        resiko ='Anda beresiko terkena gagal jantung'

    st.success(resiko)