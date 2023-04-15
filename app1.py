import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("classifier.pkl","rb")
models = pickle.load(pickle_in)

@st.cache
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    st.title("ION Channel Switching predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white; text-align:center;">ION Channel Switching Predictor App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="Signal batch input(CSV)", type=['csv'])
    batch_length = st.number_input(label="Length of each batch recording")
    total_batches = st.number_input(label="No. of Input batches")
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        data = test_data.to_numpy()
        signal = data[:,1]
        signal = np.array(np.array_split(signal, 4))
    
    if st.button("Predict") and uploaded_file is not None:
        model_ref = [0,2,4,0,1,3,4,3,0,2,0,0,0,0,0,0,0,0,0,0]
        y_pred_all = np.zeros((test_data.size))
        for pec in range(20):
            x_test = signal.flatten()[pec*100000:(pec+1)*100000]
            x_test = np.expand_dims(np.array(x_test),-1)
            test_pred = models[model_ref[pec]].predict(x_test)
            y_pred_1 = np.array(test_pred).astype(int)
            y_pred_all[pec*100000:(pec+1)*100000] = y_pred_1
        
        y_pred_all = np.array(y_pred_all).astype(int)
        st.success("Prediction is complete. Result will be downloaded as CSV file")
        result_csv = convert_df_to_csv(pd.DataFrame(y_pred_all))
        st.download_button(
            label="Download data as CSV",
            data=result_csv,
            file_name='large_df.csv',
            mime='text/csv',
        )

        
if __name__ == '__main__':
    main()