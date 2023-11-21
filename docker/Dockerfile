
# Base container image, which has the Ubuntu Linux distribution
FROM    ubuntu:20.04

# Install the following packages
RUN     apt-get update && apt-get install software-properties-common -y &&\
        add-apt-repository ppa:deadsnakes/ppa && apt-get update &&\
        apt-get install python3.10 -y && apt install python3-pip -y &&\
        pip3 install --upgrade pip &&\
        apt-get install graphviz -y && apt install libgl1-mesa-glx -y

# Install any Python modules required in the notebook
RUN     pip3 install ipython && pip3 install pandas==1.5.3 && pip3 install numpy==1.23.5 && pip3 install seaborn &&\
        pip3 install matplotlib && pip3 install scikit-learn==1.3.0 && pip3 install tensorflow &&\
        pip3 install scipy && pip3 install streamlit==1.24.0 flask==2.3.2 Flask-RESTful==0.3.10 &&\
        pip3 install shap lime 


COPY . .