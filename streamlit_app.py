import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Diamond Dataset Analysis with Streamlit")
st.image("images/streamlit.png", width=500)

data = sns.load_dataset('diamonds')

menu = st.sidebar.radio('Menu', ['Home', 'Prediction Price'])

if menu == 'Home':
    st.image('images/diamond-store.jpg', width=500)

    st.header('Tabular Data of Diamonds')
    if st.checkbox('Show First 150 Rows'):
        st.write(data.head(150))

    st.header('Statistical Summary')
    if st.checkbox('Show Summary Statistics'):
        st.write(data.describe())

    st.header('Correlation Heatmap')
    if st.checkbox('Show Correlation Heatmap'):
        # Exclude non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        correl, ax = plt.subplots(figsize=(5, 2.5))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(correl)
    
    st.header('Graphs')
    graph=st.selectbox('Differernt types of graphs',['Choose a graph','Scatter Plot', 'Bar Graph','Histogram'])
    if graph=='Scatter Plot':
        caratValue=st.slider('Carat Filter',0,6)
        scatterData=data.loc[data['carat']>=caratValue]
        scatter,ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(scatterData,x='carat',y='price',hue='cut')
        st.pyplot(scatter)

    if graph=="Bar Graph":
        bar,ax=plt.subplots(figsize=(5,5))
        sns.barplot(x='cut',y=data.cut.index,data=data)
        st.pyplot(bar)
        
    if graph=="Histogram":
        hist,ax=plt.subplots(figsize=(5,3))
        sns.histplot(data.price,kde=True)
        st.pyplot(hist)

if menu=='Prediction Price':
    st.title('Diamond Prediction Price')

    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    x=np.array(data['carat']).reshape(-1,1)
    y=np.array(data['price']).reshape(-1,1)
    lr.fit(x,y)

    values=st.number_input('Carat',0.3,5.01,step=0.1)

    value=np.array(values).reshape(-1,1) # create 2D array
    prediction=lr.predict(value)[0]
    if st.button('Predict Price'):
        st.write(f"$ {round(prediction[0],2)}")

