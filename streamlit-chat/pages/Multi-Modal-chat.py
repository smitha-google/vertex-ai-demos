import streamlit as st
from google.cloud import aiplatform
import vertexai
from pathlib import Path
import os
from smutils import multimodalutils
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

#creating session states
if "image_metadata_df" not in st.session_state:
    st.session_state['image_metadata_df'] = pd.DataFrame()
    
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

@st.cache_resource
def load_model():
    model = GenerativeModel("gemini-pro")
    multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    return model, multimodal_model

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
       # Specify the image description prompt. Change it
       image_description_prompt = """Explain what is going on in the image.
        If it's a table, extract all elements of the table.
        If it's a graph, explain the findings in the graph.
        Do not include any numbers that are not mentioned in the image.
        """
       text_metadata_df, image_metadata_df = multimodalutils.get_document_metadata(GenerativeModel("gemini-1.0-pro-vision"),  # we are passing gemini 1.0 pro vision model
                    file,file_name, image_save_dir="images",image_description_prompt=image_description_prompt,embedding_size=1408,)
       frames = [st.session_state['text_df'], text_metadata_df]
       frames1 = [st.session_state['image_metadata_df'], image_metadata_df]
       st.session_state['text_df'] = pd.concat(frames)
       st.session_state['image_metadata_df'] = pd.concat(frames1)
       st.info("File is indexed and ready for Q&A")

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
model, multimodal_model = load_model()

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
        matching_results_text = multimodalutils.get_similar_text_from_query(prompt,
            st.session_state['text_df'],column_name="text_embedding_chunk",top_n=10,
            chunk_text=True,
        )

        # Get all relevant images based on user query
        matching_results_image_fromdescription_data = multimodalutils.get_similar_image_from_query(
            st.session_state['text_df'], st.session_state['image_metadata_df'],
            query=prompt,
            column_name="text_embedding_from_image_description",  # Use image description text embedding
            image_emb=False,  # Use text embedding instead of image embedding
            top_n=10,
            embedding_size=1408,
        )

        # combine all the selected relevant text chunks
        context_text = []
        for key, value in matching_results_text.items():
            context_text.append(value["chunk_text"])
        final_context_text = "\n".join(context_text)

        # combine all the relevant images and their description generated by Gemini
        context_images = []
        for key, value in matching_results_image_fromdescription_data.items():
            #st.image(value["image_object"].data)
            context_images.extend(
            ["Image: ", value["image_object"], "Caption: ", value["image_description"]]
        )

        final_prompt = f""" Instructions: Compare the images and the text provided as Context: to answer multiple Question:
        Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.
        If unsure, respond, "Not enough context to answer".

        Context:
        - Text Context:
        {final_context_text}
        - Image Context:
        {context_images}

        {prompt}

        Answer:
        """

        response = multimodalutils.get_gemini_response(
                multimodal_model,  # we are passing Gemini 1.0 Pro Vision
                model_input=[final_prompt],
                stream=True,
                generation_config=GenerationConfig(temperature=0.2, max_output_tokens=2048),
            )
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})