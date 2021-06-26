import streamlit as st
import seaborn as sns
import pickle
import pandas as pd
from Model import a1, a2, a3, a4


lin_model = pickle.load(open('lin_model.pkl', 'rb'))
log_model = pickle.load(open('log_model.pkl', 'rb'))
svm = pickle.load(open('svc_model.pkl', 'rb'))
knn = pickle.load(open('knn_model.pkl', 'rb'))

df = pd.read_csv("plot.csv")
def py_plot():
    html_trends = """<div style="background-color:teal ;padding-bottom:2px">
    <h2 style="color:white;text-align:center;"><b>Data Visualization</b></h2>
    </div>
    <br>
    """
    st.markdown(html_trends, unsafe_allow_html=True)
    fig = sns.pairplot(df, hue="left")
    st.pyplot(fig,width=1100,height=900)

def reg_plot():
    html_trends = """<div style="background-color:teal ;padding-bottom:2px">
            <h3 style="color:white;text-align:center;"><b>Data Visualization</b></h2>
            </div>
            <br>
            """
    st.markdown(html_trends, unsafe_allow_html=True)
    st.image('./reg_plot.png', width=None)

def classify(num):
    if num == 1:
        return 'Will leave the Job'
    else:
        return 'Not leave'


def main():
    html_title = """
    <div style="background-color:black ;padding-bottom:4px">
    <h1 style="color:white;text-align:center;">Mini-Project for Summer Internship Program</h1>
    </div><br>
    """
    st.markdown(html_title, unsafe_allow_html=True)
    html_temp = """
    <div style="background-color:teal ;padding:5px">
    <h2 style="color:white;text-align:center;"><b>Resignation Prediction using different ML Algorithms</b></h2>
    </div>
    """

    html_info = """<div style="background-color:black ;padding:5px">
    <h1 style="color:white;text-align:center;"><b>Darsh Bhatt</b></h2>
    </div><br>
    <h4 style="color:white;text-align:center; background-color:teal">Enrollment No.: <i><b>180770107009</b></i><br><br>Semester: <i><b>7</b></i><br><br>Class: <i><b>A</b></i><br><br>Branch: <i><b>CE</b></i><br></h4>
    <br>
    """

    st.sidebar.image('./atom2.gif')
    st.sidebar.markdown(html_info, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Linear Regression', 'Logistic Regression', 'SVM', 'KNN']
    st.image('./PIC.jpg')
    html_select = """ <h4 style="color:teal;"><b><br>Which model would you like to use?</b>
    <br>
    """
    st.markdown(html_select, unsafe_allow_html=True)
    option = st.selectbox('', activities)
    st.subheader(option)
    sl = st.slider('Satisfaction Level:', 0.0, 10.0)
    project = st.number_input('Enter number of projects:')
    hr = st.slider('Avg day hrs:', 0.0, 24.0)
    time = st.number_input('Enter yrs in company:')
    acc = st.select_slider('Work accident:', options=['YES', 'NO'])
    sal = st.select_slider('Current Salary:', options=[
                           'LOW', 'MEDIUM', 'HIGH'])
    if acc == 'YES':
        acc = 1
    else:
        acc = 0
    if sal == 'LOW':
        sal = 1
    elif sal == 'MEDIUM':
        sal = 2
    else:
        sal = 3

    
    inputs = [[sl/10, project, hr*30, time, acc, sal]]
    if st.button('Check'):
        if option == 'Linear Regression':
            st.success(classify(lin_model.predict(inputs)))
            st.text('R2 Score')
            st.success(a2)
            st.text('R2 Score is negative as here we are using Regression for Classification problem. ')
            st.text('The regression line is a straight line. Whereas logistic regression is for classification')
            st.text(' problems, which predicts a probability range between 0 to 1.')
            st.text('We can use Logistic Regression, SVM, and KNN for this Data')
            reg_plot()
            st.text('As we can see the red straight lines in the above plot are the predicted trend lines for')
            st.text('the independent variables of the Data which we get from the Regression Model.')
            st.text('But the Dependent variable has Classification of either 1 or 0. ')
            st.error('So this model Failed for this data.')
        elif option == 'Logistic Regression':
            check = classify(log_model.predict(inputs))
            if check == 'Will leave the Job':
                st.warning(check)
            else:
                st.success(check)
            st.text('Accuracy using Logistic Regression')
            st.success(a1*100)
            py_plot()
        elif option == 'SVM':
            check = classify(svm.predict(inputs))
            if check == 'Will leave the Job':
                st.warning(check)
            else:
                st.success(check)
            st.text('Accuracy using SVM')
            st.success(a3*100)
            py_plot()
        else:
            check = classify(knn.predict(inputs))
            if check == 'Will leave the Job':
                st.warning(check)
            else:
                st.success(check)
            st.text('Accuracy using KNN')
            st.success(a4*100)
            st.success("Best Model for this Data")
            py_plot() 


if __name__ == '__main__':
    main()
