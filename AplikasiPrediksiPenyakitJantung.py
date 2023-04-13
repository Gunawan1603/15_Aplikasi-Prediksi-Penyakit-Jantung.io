import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix

# page title
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="https://e7.pngegg.com/pngimages/594/747/png-clipart-heart-heart-cartoon-heart.png",
)

# hide menu
hide_streamlit_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">', unsafe_allow_html=True)
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> Source Code</button></a></div>', unsafe_allow_html=True)

#Home Page Sidebar
st.sidebar.image("logo-upb.png")
st.sidebar.markdown('# MAIN MENU :')
home=st.sidebar.button('🏠 Home')
about=st.sidebar.button('📚 About')
#st.sidebar.radio("Plots")
# home page
if home==False and about==False or home==True and about==False:
     st.markdown("<h1 style='text-align: center; color: Blue; margin:0 ; padding:0;'>Prediksi Penyakit Jantung</h1>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('📌 home','📊 Dataframe','📈 Chart')
)

#Prediksi Penyakit Jantung

#loading dataset
df=pd.read_csv("Data Prediksi Penyakit Jantung.csv")
x=df.iloc[:,[2,3,4,7]].values
x=np.array(x)
y=y=df.iloc[:,[-1]].values
y=np.array(y)
#performing train-test split on the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model=RandomForestClassifier()
#fitting the model with train data (x_train & y_train)
model.fit(x_train,y_train)

#Prediksi Penyakit Jantung
 
st.header("Model Klasifikasi Dengan metode Random Forest")
st.write("Silahkan Masukkan Nilai Sesuai Rentang setiap Kolom:")
chestpain=st.number_input("Nilai Nyeri Dada Anda(1-4)",min_value=1,max_value=4,step=1)
bp=st.number_input("Masukkan Nilai Tekanan Darah Anda (95-200)",min_value=95,max_value=200,step=1)
cholestrol=st.number_input("Masukkan Nilai Tingkat Kolesterol Anda (125-565)",min_value=125,max_value=565,step=1)
maxhr=st.number_input("Masukkan Detak Jantung Maksimum Anda (70-200)",min_value=70,max_value=200,step=1)
#prediksi variabel1 diprediksi berdasarkan kondisi kesehatan dengan meneruskan 4 fitur ke model
prediction=model.predict([[chestpain,bp,cholestrol,maxhr]])[0]
    
if st.button("Predict"):
        if str(prediction)=="Presence":
            st.warning("Anda Mungkin Terkena Penyakit Jantung")
            st.subheader("")
            model = RandomForestClassifier( n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("**Hasil Persentasi Accuracy dari model Random Forest adalah ", accuracy.round(2),"%**")
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Note : ")
            st.write("Presence -> Mungkin **Terkena** Penyakit Jantung")
            st.write("Absence -> Mungkin **Aman** Penyakit Jantung")

        elif str(prediction)=="Absence":
            st.success("Kamu aman")
            st.subheader("")
            model = RandomForestClassifier( n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("**Hasil Persentasi Accuracy dari model Random Forest adalah ", accuracy.round(2),"%**")
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Note : ")
            st.write("Presence -> Mungkin **Terkena** Penyakit Jantung")
            st.write("Absence -> Mungkin **Aman** Penyakit Jantung")
            



# about page
if about==True and home==False:
    url = 'https://www.kaggle.com/datasets/gnwnupb/dataprediksipenyakitjantung'
    
    st.markdown("<h2 style='text-align: center; color: Red; margin:0 ; padding:0;'>Tentang Sistem ini</h2>", unsafe_allow_html=True)
    st.image("©-iStock-peterschreiber.media.jpg")
    st.write('Sistem Prediksi Penyakit Jantung adalah sebuah sistem yang bertujuan untuk memprediksi penyakit jantung dini. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit.')
    st.markdown("<h4 style='text-align: center; color: white; margin:0 ; padding:0;'>Dataset</h4>", unsafe_allow_html=True)
    st.markdown("<p  color: white;'>Dataset yang digunakan pada sistem ini memiliki <b>4 fitur</b> termasuk kelas, Dataset yang digunakan dalam sistem ini menggunakan dataset yang berada pada website Kaggle.com . Dataset yang berjudul <i>Data Prediksi Penyakit Jantung</i>, dataset untuk mendeteksi apakah seseorang mengidap Penyakit Jantung atau tidak berdasarkan berbagai faktor seperti <i>Chest Pain</i> (mg/dL),<i>Blood Pressure(BP)</i> (mm Hg),<i>Cholestrol</i>, Max Heart Rate(HR)</i>, semua fitur yang disebutkan bertipe numerik.</p>", unsafe_allow_html=True)
    
    st.info("Dataset : [link](%s)" % url,icon="ℹ️")
    
    st.markdown("<h4 style='text-align: center; color: Red; margin:0 ; padding:0;'>Tahap preprosessing</h4>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: white; margin:0 ; padding:0;'>Metode yang digunakan</h4>", unsafe_allow_html=True)
    st.write('Memakai model Random Forest dengan akurasi sebesar 70% .')
    st.write('Random Forest bekerja dalam dua fase. Fase pertama yaitu menggabungkan sejumlah N decision tree untuk membuat Random Forest. Kemudian fase kedua adalah membuat prediksi untuk setiap tree yang dibuat pada fase pertama.')
    st.image("Algoritma Random Forest.jpg")
    st.write('Cara kerja algoritma Random Forest dapat dijabarkan dalam langkah-langkah berikut:')
    st.write('1. Algoritma memilih sampel acak dari dataset yang disediakan.')
    st.write('2. Membuat decision tree untuk setiap sampel yang dipilih. Kemudian akan didapatkan hasil prediksi dari setiap decision tree yang telah dibuat.')
    st.write('3. Dilakukan proses voting untuk setiap hasil prediksi. Untuk masalah klasifikasi menggunakan modus (nilai yg paling sering muncul), sedangkan untuk masalah regresi akan menggunakan mean (nilai rata-rata).')
    st.write('4. Algoritma akan memilih hasil prediksi yang paling banyak dipilih (vote terbanyak) sebagai prediksi akhir.')

    st.markdown("<h4 style='text-align: center; color: Red; margin:0 ; padding:0;'>Penjelasan Singkat</h4>", unsafe_allow_html=True)

    st.write('Algoritma Random Forest disebut sebagai salah satu algoritma machine learning terbaik, sama seperti Naïve Bayes dan Neural Network. Random Forest adalah kumpulan dari decision tree atau pohon keputusan. Algoritma ini merupakan kombinasi masing-masing tree dari decision tree yang kemudian digabungkan menjadi satu model. Biasanya, Random Forest dipakai untuk masalah regresi dan klasifikasi dengan kumpulan data yang berukuran besar..')
    #st.info("[Percobaan model pertama](%s) | [Percobaan model Kedua](%s) | [Percobaan model Ketiga](%s) | [Percobaan model Keempat](%s)" % (n8,n9,s8,s10),icon="ℹ️")     



#menampilkan halaman utama
    #if option == 'home' or option == '':
    #st.write("""# Halaman Utama""") #menampilkan halaman utama
elif option == '📊 Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    st.markdown('**Menampilkan 5(lima) baris pertama** dataset digunakan sebagai contoh.')
    st.write(df.head(5)) #menampilkan 5(lima) baris pertama dari kumpulan data.

    st.markdown('**Menampilkan 5(lima) baris Terakhir** dataset digunakan sebagai contoh.')
    st.write(df.tail(5)) #menampilkan 5(lima) baris Terakhir dari kumpulan data.   
    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    #df = pd.DataFrame({
        #'Column 1':[1,2,3,4],
       #'Column 2':[10,12,14,16]
    #})
    #df #menampilkan dataframe
elif option == '📈 Chart':
    st.write("""## Draw Charts""") #menampilkan judul halaman 

    #membuat variabel chart data yang berisi data dari dataframe
    #data berupa angka acak yang di-generate menggunakan numpy
    #data terdiri dari 2 kolom dan 20 baris
    chart_data = pd.DataFrame(
        np.random.randn(20,2), 
        columns=['a','b']
    )
    #menampilkan data dalam bentuk chart
    st.line_chart(chart_data)
    #data dalam bentuk tabel
    chart_data

    #Cukup Sekian & TERIMAKASIH
    #Gunawan-312010191


st.info("Made in Gunawan_TI20B1_UPB,with ❤️ by Streamlit")
     
       
