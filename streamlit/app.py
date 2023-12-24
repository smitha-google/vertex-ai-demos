import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image, 
                                                HarmCategory, 
                                                HarmBlockThreshold, 
                                                Part)
from vertexai.language_models import TextGenerationModel

from google.cloud import aiplatform
import vertexai

PROJECT_ID = "smithaargolisinternal"#os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = "us-central1"#os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-pro")
    multimodal_model_pro = GenerativeModel("gemini-pro-vision")
    #text_bison_model = TextGenerationModel.from_pretrained("text-bison@002")
    return text_model_pro, multimodal_model_pro

def get_gemini_pro_text_response(model: GenerativeModel,
                                  prompt: str, 
                                  generation_config: GenerationConfig,
                                  stream=True):
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    responses = model.generate_content(prompt,
                                       generation_config = generation_config,
                                       safety_settings=safety_settings,
                                       stream=True)

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

def get_llama_model_response(prompt: str):
    #endpoint_name = endpoint_without_peft.name
    endpoint_name = "5445777738281517056"  # @param {type:"string"}
    aip_endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_name}"
        )
    endpoint_without_peft = aiplatform.Endpoint(aip_endpoint_name)

    instances = [
        {
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 10,
        },
    ]
    response = endpoint_without_peft.predict(instances=instances)
    final_response = []
    for prediction in response.predictions:
        try:
            final_response.append(prediction)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)

st.header("Smithas Cool Demos", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

tab1, tab2, tab3 = st.tabs(["Gemini", "PaLM", "Meta-Llama2"])

with tab1:
   image_path = Path(__file__).with_name("gemini.jpg").relative_to(Path.cwd())
   st.header("Gemini")
   st.image(str(image_path), width=200)
   prompt = st.text_input("Input Prompt", key="geminiPrompt")
   reply = st.button("Submit", key="geminisubmit")
   config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
   
   if reply:
       response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
        )
       
       if response:
            st.write("From Gemini Model:")
            st.write(response)
with tab2:
   image_path = Path(__file__).with_name("palm2.png").relative_to(Path.cwd())
   st.header("PaLM")
   st.image(str(image_path), width=200)
   prompt = st.text_input("Input Prompt", key="palmPrompt")
   reply = st.button("Submit", key="palmsubmit")
   parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "top_p": 1
   }
   if reply:
       response = text_bison_model.predict(prompt=prompt, **parameters)
       st.write("From Text Bison Model2:")
       st.write(response)
with tab3:
   image_path = Path(__file__).with_name("metaai.jpeg").relative_to(Path.cwd())
   st.header("Llama-2")
   #st.image("../images/metaai.jpg", width=200)
   st.image(str(image_path), width=200)
   prompt = st.text_input("Input Prompt", key="llamaPrompt")
   reply = st.button("Submit", key="llamasubmit")

   if reply:
       response = get_llama_model_response(
                    prompt,
        )
       if response:
            st.write("From Meta's Llama Model:")
            st.write(response)
   