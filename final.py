import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from sklearn.model_selection import train_test_split

st.markdown(f"<h1 style='text-align: center; color: green;animation: fadeIn 2s;'>Shaila's Stock prediction</h1>", unsafe_allow_html=True)
st.markdown(f"<h6 style='text-align: center; color: maroon;animation: fadeIn 2s;'>Welcome to the Stock Price Prediction App, your go-to tool for forecasting stock prices and making informed investment decisions. </h6>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center; color: silver;animation: fadeIn 2s;'>Shaila Hossain Nodi</h5>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center; color: silver;animation: fadeIn 2s;'>ID:201-15-13715</h5>", unsafe_allow_html=True)
st.write()


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
st.write("data from 2008-2022",X.describe())
st.subheader('Opening price vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(X.Open)
st.pyplot(fig)
st.write("data from 2008-2022",X.describe())
st.subheader('closing price vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(X.Close)
st.pyplot(fig)


def get_Algorithm(Algorithm):
    if Algorithm=="Linear Regression":
        algo=X

        algo["Date"] = pd.to_datetime(algo["Date"])

        feat = algo[['Open', 'High', 'Low']]
        dep= algo['Close']

        feat_train, feat_test, dep_train, dep_test = train_test_split(feat, dep, random_state=0)

        regressor = LinearRegression()
        regressor.fit(feat_train, dep_train)
        predicted = regressor.predict(feat_test)

        df = pd.DataFrame({'Actual Price': dep_test, 'Predicted price': predicted})

        df['Date'] = algo['Date']
        df = df.sort_values(by='Date')

        plt.style.use('dark_background')
        fig2=plt.figure(figsize=(10,6))
        plt.plot(df['Date'], df['Actual Price'], label='Actual Price',color='red')
        plt.plot(df['Date'], df['Predicted price'], label='Predicted Price',color='green')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual Price vs Predicted Price')
        plt.legend()
        st.pyplot(fig2)
        regressor.score(feat_test,dep_test)*100

        last_30_days_data = algo.tail(30)
        last_30_days_features = last_30_days_data[['Open', 'High', 'Low']]
        last_30_days_target = last_30_days_data['Close']
        next_day_price = regressor.predict(last_30_days_features.iloc[-1].values.reshape(1, -1))
        st.write("Predicted Next Day's Stock Price:", next_day_price[0])

    elif Algorithm=="SVM":
        algo=X

        algo["Date"] = pd.to_datetime(algo["Date"])

        feat = algo[['Open', 'High', 'Low']]
        dep= algo['Close']

        feat_train, feat_test, dep_train, dep_test = train_test_split(feat, dep, random_state=0)
        svm = SVR()
        svm.fit(feat_train, dep_train)
        predicted = svm.predict(feat_test)
        df = pd.DataFrame({'Actual Price': dep_test, 'Predicted price': predicted})
        df['Date'] = algo['Date']
        df = df.sort_values(by='Date')
        plt.style.use('dark_background')
        fig3=plt.figure(figsize=(10,6))
        plt.plot(df['Date'], df['Actual Price'], label='Actual Price',color='blue')
        plt.plot(df['Date'], df['Predicted price'], label='Predicted Price',color='yellow')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual Price vs Predicted Price')
        plt.legend()
        st.pyplot(fig3)
        svm.score(feat_test,dep_test)*100

        last_30_days_data = algo.tail(30)
        last_30_days_features = last_30_days_data[['Open', 'High', 'Low']]
        last_30_days_target = last_30_days_data['Close']
        next_day_price = svm.predict(last_30_days_features.iloc[-1].values.reshape(1, -1))
        st.write("Predicted Next Day's Stock Price:", next_day_price[0])
    else:
        algo=X
        plt.style.use('dark_background')
        fig1=plt.figure(figsize=(16,8))
        plt.title('Closing Price')
        plt.plot(algo['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        st.pyplot(fig1)

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


        def calculate_mape(y_true, y_pred):
          return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Inverse transform the y_test values
        y_test_inverse = scaler.inverse_transform(y_test)

# Inverse transform the predictions
        predictions_inverse = scaler.inverse_transform(predictions)

# Calculate MAPE
        mape = calculate_mape(y_test_inverse, predictions_inverse)

        st.write(f'Accuracy: {100-mape:.2f}%')





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
        plt.plot(train['Close'] , color='red')
        plt.plot(valid['Close'] , color='yellow')
        plt.plot(valid[ 'predictions'] , color='green')
        plt.legend(['Train','Validation', 'predictions'], loc='lower right')
        st.pyplot(fig2)

        new_df=algo.filter(['Close'])
#Get last 60 days values and convert into array
        last_60_days=new_df[-60:].values
        scaler = StandardScaler()  # Or any other scaler
        scaler.fit(new_df)
 #Scale the data to be values between 0
        last_60_days_scaled=scaler.transform(last_60_days)

#Create an empty list
        X_test=[]
#Appemd the past 60days
        X_test.append(last_60_days_scaled)

#Conver the X_test data into numpy array
        X_test = np.array(X_test)

#Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#Get predicted scaled price
        pred_price = model.predict(X_test)
        pred_price = np.reshape(pred_price, (pred_price.shape[0], 1))
#undo the scaling
        pred_price=scaler.inverse_transform(pred_price)
        st.write(f'Price of tomorrow:{pred_price}')

        valid.tail(15)
    return algo
algo=get_Algorithm(Algorithm)


