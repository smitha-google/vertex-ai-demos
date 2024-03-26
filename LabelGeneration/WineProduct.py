import streamlit as st
from google.cloud import aiplatform
import vertexai
from pathlib import Path
import os
import numpy as np
import pandas as pd
import base64
from vertexai.preview.generative_models import (GenerativeModel, Part)
import vertexai.preview.generative_models as generative_models
from PIL import Image
import io


PROJECT_ID = "smithaargolisinternal"#os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = "us-central1"#os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
st.title("Product Description Generation")
model = GenerativeModel("gemini-1.0-pro-vision")

if 'disabled' not in st.session_state:
    st.session_state.disabled = False

def disable():
    st.session_state.disabled = True

def generate(image, prompt):
  responses = model.generate_content(
    [image, prompt],
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    stream=True,
  )
  return responses

with st.form("my_form"):
    uploaded_images = st.file_uploader('Choose your .jpeg/.png file', type=['png', 'jpg'],accept_multiple_files=True, 
                                disabled=st.session_state.disabled)
    prompt = st.text_area(
    "Enter your prompt",
    "Generate the product description of the images uploaded and suggest pairings",
    )
    final_prompt = f""" Instructions: You are wine expert. Always return the name of the wine. The voice has to be fairly authoritative or crisp. 
    Do not use flowery or descriptive language.

        {prompt}

        Answer:
        """

    submitted = st.form_submit_button("Submit", on_click=disable())
    st.spinner("Generating")
    for uploaded_image in uploaded_images:
        if submitted:
           st.info("successfully uploaded!")
           image = Image.open(uploaded_image)
           img_byte_arr = io.BytesIO()
           image.save(img_byte_arr, format='PNG')
           st.write(image.filename)
           img_byte_arr = img_byte_arr.getvalue()
           col1, col2 = st.columns([0.2,0.8])
           with col1:
              st.image(image, caption=image.filename, width=200) 
           with col2:
              responses = generate(Part.from_data(img_byte_arr, mime_type="image/png"), prompt)
              for response in responses:
                 col2.write(response.text)