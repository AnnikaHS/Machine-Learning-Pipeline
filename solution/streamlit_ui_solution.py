import streamlit as st
from workshop.pipeline import Pipeline

st.title("Your first ML Data app using streamlit :) ")

st.markdown("""
## Task 10 (and last!)

Call your favourite model using mlflow from withing streamlit and visualise it in the streamlit web app.

Tip: to run and visualize streamlit run in the terminal:
```
streamlit run workshop/streamlit_ui.py
```
            
## Solution
        """)
pipeline = Pipeline()
content = st.text_input("Type your query", value='I lost my card')
output = pipeline.predict_mlflow_model(content)
if st.button("Predict"):
    st.text(output)
