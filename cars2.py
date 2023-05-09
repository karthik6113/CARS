import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
logo = Image.open('logo1.png')

#pip install pandas numpy matplotlib seaborn streamlit
#to run strealit :   streamlit run test2.py 
st.set_page_config(page_title=" CARS EDA", page_icon=":bar_chart:", layout="wide")
st.image(logo)
st.title("𝙴𝚇𝙿𝙻𝙾𝚁𝙰𝚃𝙾𝚁𝚈 𝙳𝙰𝚃𝙰 𝙰𝙽𝙰𝙻𝚈𝚂𝙸𝚂 (𝙴𝙳𝙰)")
# File upload
uploaded_file = st.file_uploader("𝚄𝙿𝙻𝙾𝙰𝙳 𝙰 𝙲𝙰𝚁𝚂 𝙳𝙰𝚃𝙰 𝚂𝙴𝚃 :")
if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.dataframe(data)
    names = ["𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟷𝟹 – 𝙶 𝙺 𝙱𝙷𝙰𝚂𝙺𝙰𝚁","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟹𝟹 – 𝙻 𝙷𝙰𝚁𝚂𝙷𝙰 𝚅𝙰𝚁𝙳𝙷𝙰𝙽","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟶𝟾 – 𝙶 𝚂𝚆𝙰𝚁𝚄𝙿𝙰","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟸𝟶 – 𝙺 𝙿𝚁𝙰𝚂𝙰𝙳","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟸𝟾 – 𝙺 𝚂𝙷𝚈𝙰𝙼𝙰𝙻𝙰","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟷𝟶 – 𝙳 𝚂𝙷𝚈𝙰𝙼","𝟸𝟷𝙰𝟸𝟷𝙰𝟼𝟷𝟸𝟸 – 𝙺 𝚄𝙼𝙰 𝚂𝙰𝙸","𝟸𝟶𝙰𝟸𝟷𝙰𝟼𝟷𝟶𝟾 – 𝙱 𝚂𝙰𝙸 𝚁𝙰𝙼"]
    st.sidebar.title("𝙿𝚁𝙾𝙹𝙴𝙲𝚃 𝚃𝙴𝙰𝙼 𝙼𝙴𝙼𝙱𝙴𝚁𝚂 :")

    for name in names:
        st.sidebar.write(name)

    st.sidebar.title("𝚄𝙽𝙳𝙴𝚁 𝚃𝙷𝙴 𝙶𝚄𝙸𝙳𝙴𝙽𝙲𝙴 𝙾𝙵 :")
    st.sidebar.write(" 𝙳𝚁 𝙱𝙾𝙼𝙼𝙰 𝚁𝙰𝙼𝙺𝚁𝙸𝚂𝙷𝙽𝙰")
     
   
    st.title("𝙲𝙰𝚁𝚂 𝙳𝙰𝚃𝙰 𝙰𝙽𝙰𝙻𝚈𝚂𝙸𝚂")
    if st.checkbox("RAW DATA INFO :"):
      st.write("NULL VALUES IN EACH COLUMN :")
      st.write(data.isnull().sum())
    if st.checkbox("DATA INFO AFTER CLEANING :"):
      data=data.dropna()
      st.write("NULL VALUES IN EACH COLUMN :")
      st.write(data.isnull().sum())   
       
    if st.checkbox("𝚂𝙷𝙾𝚆 𝙵𝙸𝚁𝚂𝚃 𝟸𝟻 𝚁𝙾𝚆𝚂 :"):
        st.write(data.head(25))
    if st.checkbox("STASTICAL OBSERVATIONS ON DATASSET :"):
        st.write(data.describe())

    if st.checkbox(" 𝚂𝙷𝙾𝚆 𝚂𝙷𝙰𝙿𝙴 AND DIMENSIONS OF DATAFRAME :"):
        st.write("SHAPE :",data.shape)
        st.write(" NO OF DIMENSIONS :",data.ndim)
        
    if st.checkbox("BASIC DETAILS ABOUT MANUFACTURERS :"):
        data=data.dropna()
        if st.checkbox("DIFFERENT TYPES OF MANUFACTURERS AND THIER COUNTS :"):
          st.write(data['Make'].value_counts())

        if st.checkbox("𝙼𝙰𝙽𝚄𝙵𝙰𝙲𝚃𝚄𝚁𝙴𝚁  𝚆𝙷𝙾 𝙿𝚁𝙾𝙳𝚄𝙲𝙴𝚂 𝚂𝙴𝙳𝙰𝙽 𝙼𝙾𝙳𝙴𝙻 𝙲𝙰𝚁𝚂 :"):
            data1=data.where(data['Type']=='Sedan')['Make'].unique()
            st.write(data1[1:])

        if st.checkbox("𝙼𝙾𝙳𝙴𝙻 𝚃𝚈𝙿𝙴 𝚃𝙾 𝙲𝙷𝙴𝙲𝙺 𝚃𝙷𝙴 𝙼𝙰𝙽𝚄𝙵𝙰𝙲𝚃𝚄𝚁𝙴𝚁𝚂 :"):
            st.write("PLEASE CHOOSE MODELS FROM THE BELOW LIST :",data['Type'].unique())
            x=st.text_input("PLEASE ENTER MODEL TYPE :")
            data1=data.where(data['Type']==x)['Make'].unique()
            st.write(data1[1:])

        if st.checkbox("SHOW ALL THE RECORDS ARE ORIGIN IN EUROPE OR ASIA"):
          st.write(data[data['Origin'].isin(['Asia', 'Europe'])])
          st.write(data['Origin'].value_counts())

    if st.checkbox("SHOW CORRELATION BETWEEN MPG_CITY AND MPG_HIGHWAY :"):
        st.write(data['MPG_City'].corr(data['MPG_Highway']))

    if st.checkbox("SHOW SOME BASIC DETAILS OF CARS :"):
        data=data.dropna()

        if st.checkbox("PRICING DETAILS OF SELECTED MODELS :"):
            st.write("choose model and type from the below list :")
            st.write(data['Type'].unique())
            data["MSRP"] = data["MSRP"].replace("[$,]", "", regex=True).astype(int)
            y=st.text_input("PLEASE ENTER MODEL TYPE:",key=" MODEL_INPUT")
            st.write("MINIMUM VALUE IN THE SELECTED TYPE :")
            data1 = data.where(data['Type']==y)['MSRP']
            st.write(data1.min())
            st.write("MAXIMUM VALUE IN THE SELECTED TYPE :")
            data1 = data.where(data['Type']==y)['MSRP']
            st.write(data1.max())
            st.write("AVERAGE  VALUE IN THE SELECTED TYPE :")
            data1 = data.where(data['Type']==y)['MSRP']
            st.write(data1.mean())

        if st.checkbox("SHOW CARS WITH HORSPOWER IN BETWEEN 350 TO 450 :"):
           data1 = data[(350 <= data['Horsepower']) & (data['Horsepower'] <= 450)]
           st.write(data1[data.columns[:3]])

        if st.checkbox("SHOW THE CARS WITH ENGINE SIZE 3.5 AND WITH 6 CYLINDERS :"):
            data1 = data[(data['EngineSize']== 3.5 )& (data['Cylinders']== 6)]
            st.write(data1['Model'])

        if st.checkbox("SHOW CARS WITH SELECTED HORSEPOWER AND NO OF CYLINDERS"):
            col1, col2 = st.columns(2)
            slider1=data['Horsepower'].unique()
            slider2=data['Cylinders'].unique()
            with col1:
                value1 = st.selectbox('CLICK ON A VALUE OF HORSEPOWER ', slider1)
            with col2:
                value2 = st.selectbox('CLICK ON ANY NO OF CYLINDERS ',slider2 )
            data1 = data[(data['Horsepower']==value1) & (data['Cylinders']==value2)]
            st.write(data1[data.columns[:7]])
        if st.checkbox("SHOW CARS WITH SELECTED ENGINESIZE"):
            value1 = st.slider('Slide to required value', min_value=0.0, max_value=10.0, step=0.1)
            data1 = data[(data['EngineSize']==value1)]
            st.write(data1[data.columns[:3]])

    if st.checkbox("SHOW DATA VISULIAZATIONS "):
        if st.checkbox("HISTOGRAM THAT SHOWS COUNT OF DIFFERENT MAKES "):
            fig, ax = plt.subplots(figsize = (30,10))
            sns.histplot(data=data, x="Make", ax=ax)
            ax.set_xlabel("MAKE")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        if st.checkbox("DISPLAY SCATTER PLOT"):
            fig, ax = plt.subplots(figsize = (20,10))
            sns.scatterplot(data=data, x='EngineSize',y='Horsepower', ax=ax)
            plt.title("SCATTER PLOT MSRP VS INVOICE")
            ax.set_xlabel("MSRP")
            ax.set_ylabel("Invoice")
            st.pyplot(fig)
        if st.checkbox("SHOW HEATMAP OF DATA CORRELATION"):
            heatmap=sns.heatmap(data.iloc[:,[7,8,9,10]].corr())
            plt.title("HEATMAP OF DATA CORRELATION")
            st.pyplot(heatmap.figure)
        if st.checkbox("SHOW BOX PLOT OF DATA "):
            fig, ax = plt.subplots(figsize=(8, 6))
            boxplot = sns.boxplot(data=data,ax=ax)
            st.pyplot(fig)
        if st.checkbox("SHOW PIE CHART OF ORIGIN "):
            fig, ax = plt.subplots()
            x=data['Origin'].value_counts()
            mylabels = ['ASIA','USA','EUROPE']
            ax.pie(x,labels = mylabels)
            plt.title('PIE CHART OF PRODUCTION OF CARS IN EVERY COUNTRIES')
            ax.axis('equal')
            st.pyplot(fig)
        if st.checkbox("SHOW LINE PLOT"):
            fig, ax = plt.subplots()
            subset = data.loc[(data['MPG_City'] >= 0) & (data['MPG_City'] <= 12)]
            ax.plot(subset['MPG_City'], subset['MPG_Highway'], 'o-', linewidth=2, markersize=10, label='Line Label')
            ax.set_title('LINE PLOT BETWEEN MPG_CITY AND MPG_HIGHWAY')
            ax.set_xlabel('MPG_City')
            ax.set_ylabel('MPG_Highway')
            ax.legend()
            st.pyplot(fig)
    if st.checkbox("CHECK CARS ON YOUR PERSONLIZATION"):
        col1, col2,col3,col4 = st.columns(4)
        slider1=data['Make'].unique()
        slider2=data['Type'].unique()
        slider4=data['Horsepower'].unique()
        
        with col1:
            value1 = st.selectbox('CLICK ON A VALUE OF HORSEPOWER ', slider1)
        with col2:
            value2 = st.selectbox('CLICK ON ANY NO OF CYLINDERS ',slider2 )
        with col3:
            min_price = st.slider("Minimum Price", min_value=0.0, max_value=10000.0,step=100.0)
            max_price = st.slider("Maximum Price", min_value=0.0, max_value=10000.0, step=100.0)
        with col4:
            value4= st.selectbox('CLICK ON ANY NO OF Horsepower ',slider4)

        data1 = data[(data['Make']==value1) & (data['Type']==value2)&(data[(min_price<= data['Horsepower']) & (data['MSRP'] <=max_price)])&(data['Horsepower']==value4)]
        st.write(data1[data[['Make','Type','MSRP','Horsepower']]])



    
    
      
