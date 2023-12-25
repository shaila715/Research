import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from sklearn.model_selection import train_test_split

st.markdown(f"<h1 style='text-align: center; color: green;animation: fadeIn 2s;'>Shaila's Stock prediction</h1>", unsafe_allow_html=True)

dataset_name= st.sidebar.selectbox("Select dataset",("Grameenphone","Brac","Beximco"))
Algorithm=st.sidebar.selectbox("select Algorithm",("Linear Regression","SVM","Lstm"))


def get_dataset(dataset_name):
    if dataset_name=="Grameenphone":
        data=pd.read_excel("C:\\Users\\User\\Desktop\\pyproject\\Grameenphone.xlsx")
    elif dataset_name=="Brac":
        data=data2=pd.read_excel("C:\\Users\\User\\Desktop\\pyproject\\Brac.xlsx")
    else:
        data=data3=pd.read_excel("C:\\Users\\User\\Desktop\\pyproject\\Beximco.xlsx")

    X=data.copy()
    return X
X =get_dataset(dataset_name)
st.write("Shape of dataset",X.shape)

def get_Algorithm(Algorithm):
    if Algorithm=="Linear Regression":
        algo=LinearRegression()
    elif Algorithm=="SVM":
        algo=DecisionTreeClassifier()
    else:
        algo=X
        algo["Date"] = pd.to_datetime(algo["Date"])
        plt.style.use('dark_background')
        fig1=plt.figure(figsize=(16,8))
        plt.title('Opening Price')
        plt.plot(algo['Open'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Open Price', fontsize=18)
        st.pyplot(fig1)

        plt.style.use('dark_background')
        fig1=plt.figure(figsize=(16,8))
        plt.title('Closing Price')
        plt.plot(algo['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        st.pyplot(fig1)

        st.subheader('Dataset')
        st.write(algo.head(5))

        st.subheader('data from 2010-2022')
        st.write(algo.describe())

        data=algo.filter(['Close'])
        dataset=data.values
        training_data_len = math.ceil(len(dataset)*.8)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)


        train_data = scaled_data[0:training_data_len,:]
        x_train=[]
        y_train=[]

        for i in range(60, len(train_data)):
          x_train.append(train_data[i-60:i,0])
          y_train.append(train_data[i,0])


        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model=Sequential()
#Add first layer to model
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
#Add second layer to model
        model.add(LSTM(50, return_sequences=False))
#Add Dense Layer to model with 25 neurons
        model.add(Dense(25))
#Add Dense Layer to model with 1 neuron
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=64,epochs=10 )

        test_data=scaled_data[training_data_len-60: , : ]
        x_test=[]
        y_test=dataset[training_data_len:,:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i,0])
        x_test=np.array(x_test)
        x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

        predictions=model.predict(x_test)
        predictions=scaler.inverse_transform(predictions)

        rmse =np.sqrt(np.mean(predictions-y_test)**2)


        st.subheader('Prediction')
        plt.style.use('dark_background')
        train=data[:training_data_len]
        valid=data[training_data_len:]
        valid['predictions'] = predictions

#Visualize the data
        fig2=plt.figure(figsize=(16,8))
        plt.title('Model prediciton results')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(train['Close'] , color='red',linewidth=3)
        plt.plot(valid['Close'] , color='yellow',linewidth=3)
        plt.plot(valid[ 'predictions'] , color='blue',linewidth=3)
        plt.legend(['Train','Validation', 'predictions'], loc='lower right')
        st.pyplot(fig2)

        valid.tail(15)
    return algo
algo=get_Algorithm(Algorithm)

