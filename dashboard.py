import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
from Preprocessing import PreprocessingPipeline
import json
import altair as alt

st.set_page_config(layout="wide")



# Header
st.write('# Diabetic Retinopathy Diagnosis Prediction')
st.write('A model based on Convolutional Neural Network and Topological Data Analysis')

# Load model
model_path = st.text_input("Model path", None)
model = None
if model_path is not None:
    model = load_model(model_path+'/model.h5')
    with open(model_path+'/parameters.json') as f:
        params = json.load(f)
    st.write('Model loaded!')
st.write('---')

# Body
col1, col3 = st.columns(2, gap='large')

# Upload image section
col1.write('## Upload image')
uploaded_file = col1.file_uploader("Choose a file")

if uploaded_file is not None:

    # Raw image section    
    col11, col12 = col1.columns(2, gap='large')
    bytes_data = uploaded_file.getvalue()
    col11.write('Fundus image')
    col11.image(bytes_data, use_column_width=True)

    col12.text('')
    col12.text('')
    col12.text('')
    if col12.button('Make prediction!', type='primary'):

        if model is None:
            col12.write('Please input model path')

        # Run preprocessing
        with open(f'data/image_inf/{uploaded_file.name}','wb') as f:
            f.write(uploaded_file.read())
        data_info = pd.DataFrame({
            'id_code': uploaded_file.name.split('.')[0],
            'diagnosis': 0,
        }, index=[0])
        img_gen_val = ImageDataGenerator(
            rescale=1./255, 
            )
        val_gen = PreprocessingPipeline(
            data_info, 
            len(data_info), 
            img_gen_val, 
            'data/image_inf/',
            topo_channel=params['topo_channel'],
            )
        data_preprocessed = val_gen.__getitem__(0)
        
        # Run prediction
        preds = model.predict(data_preprocessed[0])
        preds[preds==0] = 0.1
        preds = pd.DataFrame(preds, columns=['Healthy', 'Mild', 'Moderate', 'Severe', 'Proliferative'])

        # Prediction result section
        col3.write('## Prediction')
        col3.write('Diagnosis Prediction')
        chart = (
            alt.Chart(preds.melt())
            .mark_bar()
            .encode(
                alt.X("value", axis=alt.Axis(title='Probability')),
                alt.Y("variable", axis=alt.Axis(title=None)),
                alt.Color(
                    "variable",
                    legend=None, 
                    scale=alt.Scale(
                        domain=preds.melt()['variable'].tolist(),
                        range=['#44ce1b','#bbdb44','#f7e379','#f2a134','#e51f1f']
                    )),
            )
            .interactive()
            .properties(
                # width=200,
                height=300
            )
        )
        col3.altair_chart(chart, use_container_width=True,)

        # Show betti curve
        col3.write('Betti Curve')
        betti_curve = pd.DataFrame({
            'Dim-0': data_preprocessed[0][1][0][:100],
            'Dim-1': data_preprocessed[0][1][0][100:],
        })
        col3.line_chart(
            betti_curve, 
            height=200,
            use_container_width=True,
            )