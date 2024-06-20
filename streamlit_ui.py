import streamlit as st
from workshop.pipeline import Pipeline

st.sidebar.write("\n\n\n\n\n")
st.sidebar.subheader("Hello 👋") 
st.sidebar.write("I am [**Annika Heintz-Saad**](https://www.linkedin.com/in/annika-heintz-saad-79791b72/), an emerging data scientist.")
st.sidebar.write("I was only able to build this ML pipeline thanks to [Girls in Tech](https://www.linkedin.com/company/girls-in-tech-germany/) and [Morena Bastiaansen](https://www.linkedin.com/in/morena-bastiaansen-225b6518/). Thanks for the cool Workshop!")

st.title("My first Machine Learning Pipeline using Natural Language Processing (NLP)")
st.write("The goal was to build a machine learning model which can predict the intent for a previously unseen user query. Each query needs to be assigned to a single intent which makes this problem a multi-class text classification.")
st.write("- I analysed and preprocessed the dataset called [Banking77](https://huggingface.co/datasets/PolyAI/banking77)")
st.write(" - I  converted texts to their vector representation to prepare the data for model training.")
st.write ("- I tried 2 simple models and selected the best parameters to use for them.")
st.write(" - In the end, I created a simple online API with Streamlit to test the solution live.") 

pipeline = Pipeline()
content = st.text_input("Try it out and type your query", value='I lost my card')
output = pipeline.predict_mlflow_model(content)
if st.button("Predict"):
    st.text(output)
