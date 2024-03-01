import streamlit as st
from google.cloud import aiplatform
import vertexai
from pathlib import Path
import os
import utils
import numpy as np
import pandas as pd
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image, 
                                                HarmCategory, 
                                                HarmBlockThreshold, 
                                                Part)
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "smithaargolisinternal"#os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = "us-central1"#os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
st.title("Chat Bot")

#creating session states
if "text_df" not in st.session_state:
    st.session_state['text_df'] = pd.DataFrame()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if 'disabled' not in st.session_state:
    st.session_state.disabled = False

def disable():
    st.session_state.disabled = True

with st.form("my_form"):
    uploaded_pdf = st.file_uploader('Choose your .pdf file', type="pdf", 
                                disabled=st.session_state.disabled)
    submitted = st.form_submit_button("Submit", on_click=disable())
    if submitted:
       st.info("successfully uploaded!")
       temp_file = "./temp.pdf"
       with open(temp_file, "wb") as file:
        file.write(uploaded_pdf.getvalue())
        file_name = uploaded_pdf.name
       st.info("Indexing and Generating Embeddings")
       text_metadata_df = utils.get_text_metadata(file, PROJECT_ID)
       frames = [st.session_state['text_df'], text_metadata_df]
       st.session_state['text_df'] = pd.concat(frames)
       st.info("File is indexed and ready for Q&A")

@st.cache_resource
def load_model():
    model = GenerativeModel("gemini-pro")
    text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    return model

def multiturn_generate_content(model: GenerativeModel,
                                  prompt: str, 
                                  generation_config: GenerationConfig,
                                  stream=True):
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    chat = model.start_chat()
    responses = chat.send_message(prompt, generation_config = generation_config,
                                      safety_settings=safety_settings,
                                      stream=True)

    final_response = []
    for response in responses:
        try:
            #st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

#Load the Gemini Model
model = load_model()

#Set the Config
config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }

#Start building the app
if prompt := st.chat_input("Add your prompt:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response=f"Echo: {prompt}"
    with st.spinner("Generating..."):
        matching_results_chunks_data = utils.get_similar_text_from_query(
            PROJECT_ID,prompt,st.session_state['text_df'],column_name="text_embedding_chunk",
            top_n=3, embedding_size=128, chunk_text=True,
        )

        context_text = []
        for key, value in matching_results_chunks_data.items():
            context_text.append(value["chunk_text"])
        #final_context_text = '\n'.join(context_text)
        #joined_string = ' '.join([str(item) for item in list_of_strings])
        final_context_text = "\n".join([str(item) for item in context_text])

        instructions = """The context of extraction of detail should be based on the text context given in "text_context": \n
        Base your response on "text_context". Do not include any cumulative total return in the answer. Context:
        """
        final_prompt = [
            prompt,
            instructions,
            "text_context:",
            "\n".join(final_context_text)
        ]
        response = multiturn_generate_content(
            model,
            final_prompt,
            generation_config=config,
        )
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})