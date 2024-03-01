from google.protobuf import struct_pb2
import fitz
import typing
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import os
from google.cloud import aiplatform
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image, 
                                                HarmCategory, 
                                                HarmBlockThreshold, 
                                                Part)
import vertexai
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel

#text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding"
)
#Load the document
#def load_doc(path: str):
def load_doc(file:str):
    # Open the PDF file
    doc: fitz.Document = fitz.open(file)

    # Get the number of pages in the PDF file
    num_pages: int = len(doc)

    return doc, num_pages

def get_text_embedding_from_text_embedding_model(
    project_id: str, text: str, embedding_size: int = 128
) -> list:
    """
    Returns an embedding (as a list) based on input text, using a multimodal embedding modal:
    multimodalembedding@001

    Args:
        project_id: The ID of the project containing the multimodal embedding model.
        text: The text to generate an embedding for.
        embedding_size: The size of the embedding vector. Defaults to 128.

    Returns:
        A list representing the text embedding.
    """

    # Create a client to interact with the Vertex AI Prediction Service
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )

    # Specify the endpoint of the deployed multimodal embedding model
    endpoint_multimodalembedding = "projects/smithaargolisinternal/locations/us-central1/publishers/google/models/multimodalembedding@001"

    # Construct an instance to represent the text input
    instance = struct_pb2.Struct()
    if text:
        instance["text"] = text

    instances = [instance]

    # Set the embedding size parameter
    parameters = {"dimension": 1408}

    # Send the prediction request and get the embedding
    response = client.predict(
        endpoint=endpoint_multimodalembedding,
        instances=instances,
        parameters=parameters,
    )
    text_embedding = [v for v in response.predictions[0].get("textEmbedding", [])]

    return text_embedding

def get_page_text_embedding(
    project_id: str, text_data: typing.Union[dict, str], embedding_size: int = 128
) -> dict:
    """
    * Generates embeddings for each text chunk using a specified embedding model.
    * Takes a dictionary of text chunks and an embedding size as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding embeddings.

    Args:
        text_data: Either a dictionary of pre-chunked text or the entire page text.
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A dictionary where keys are chunk numbers or "text_embedding" and values are the corresponding embeddings.

    """
    embeddings_dict = {}

    if isinstance(text_data, dict):
        # Process each chunk
        # print(text_data)
        for chunk_number, chunk_value in text_data.items():
            text_embd = get_text_embedding_from_text_embedding_model(
                project_id=project_id, text=chunk_value, embedding_size=128
            )
            embeddings_dict[chunk_number] = text_embd
    else:
        # Process the first 1000 characters of the page text
        text_embd = get_text_embedding_from_text_embedding_model(
            project_id=project_id, text=text_data[:1000], embedding_size=128
        )
        embeddings_dict["text_embedding"] = text_embd

    return embeddings_dict

def get_text_overlapping_chunk(
    text: str, character_limit: int = 1000, overlap: int = 100
) -> dict:
    """
    * Breaks a text document into chunks of a specified size, with an overlap between chunks to preserve context.
    * Takes a text document, character limit per chunk, and overlap between chunks as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding text chunks.

    Args:
        text: The text document to be chunked.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).

    Returns:
        A dictionary where keys are chunk numbers and values are the corresponding text chunks.

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Initialize variables
    chunk_number = 1
    chunked_text_dict = {}

    # Iterate over text with the given limit and overlap
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # Encode and decode for consistent encoding
        chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
            "utf-8", "ignore"
        )

        # Increment chunk number
        chunk_number += 1

    return chunked_text_dict

def get_chunk_text_metadata(
    project_id: str,
    page: fitz.Page,
    character_limit: int = 1000,
    overlap: int = 100,
    embedding_size: int = 128,
) -> tuple[str, dict, dict, dict]:
    """
    * Extracts text from a given page object, chunks it, and generates embeddings for each chunk.
    * Takes a page object, character limit per chunk, overlap between chunks, and embedding size as input.
    * Returns the extracted text, the chunked text dictionary, and the chunk embeddings dictionary.

    Args:
        page: The fitz.Page object to process.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A tuple containing:
            - Extracted page text as a string.
            - Dictionary of embeddings for the entire page text (key="text_embedding").
            - Dictionary of chunked text (key=chunk number, value=text chunk).
            - Dictionary of embeddings for each chunk (key=chunk number, value=embedding).

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """
    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the page
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

    # Get whole-page text embeddings
    page_text_embeddings_dict: dict = get_page_text_embedding(
        project_id, text, embedding_size
    )

    # Chunk the text with the given limit and overlap
    chunked_text_dict: dict = get_text_overlapping_chunk(text, character_limit, overlap)
    # print(chunked_text_dict)

    # Get embeddings for the chunks
    chunk_embeddings_dict: dict = get_page_text_embedding(
        project_id, chunked_text_dict, embedding_size
    )
    # print(chunk_embeddings_dict)

    # Return all extracted data
    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict

def get_text_metadata(pdf_path, PROJECT_ID):
        text_metadata: Dict[Union[int, str], Dict] = {}
        #doc_path = Path(__file__).with_name("Google_Responsibility_Report.pdf").relative_to(Path.cwd())
        doc, num_pages = load_doc(pdf_path) 
        for page_num in range(0,4):
            print(f"Processing page: {page_num + 1}")
            page = doc[page_num]
            text = page.get_text()
            (
                text,
                page_text_embeddings_dict,
                chunked_text_dict,
                chunk_embeddings_dict,
            ) = get_chunk_text_metadata(PROJECT_ID, page, embedding_size=128)
            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embeddings_dict,
                "chunked_text_dict": chunked_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict,
            }
        text_metadata_df = get_text_metadata_df(pdf_path, text_metadata)
        return text_metadata_df

def get_text_metadata_df(
    filename: str, text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a text metadata dictionary as input,
    iterates over the text metadata dictionary and extracts the text, chunk text,
    and chunk embeddings for each page, creates a Pandas DataFrame with the
    extracted data, and returns it.

    Args:
        filename: The filename of the document.
        text_metadata: A dictionary containing the text metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted text, chunk text, and chunk embeddings for each page.
    """
    final_data_text: List[Dict] = []

    for key, values in text_metadata.items():
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = key + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"][
                "text_embedding"
            ]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

            final_data_text.append(data)

    return_df = pd.DataFrame(final_data_text)
    return_df = return_df.reset_index(drop=True)
    return return_df

def get_user_query_text_embeddings(
    project_id: str, user_query: str, embedding_size: int
) -> np.ndarray:
    """
    Extracts text embeddings for the user query using a text embedding model.

    Args:
        project_id: The Project ID of the embedding model.
        user_query: The user query text.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query text embedding.
    """

    return get_text_embedding_from_text_embedding_model(
        project_id, user_query, embedding_size=128
    )

def get_cosine_score(
    dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query embedding and the dataframe embedding.
    """

    text_cosine_score = round(np.dot(dataframe[column_name], input_text_embd), 2)
    return text_cosine_score

def get_similar_text_from_query(
    project_id: str,
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "",
    top_n: int = 3,
    embedding_size: int = 128,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from a metadata DataFrame based on a text query.

    Args:
        project_id: The Project ID of the embedding model used for text comparison.
        query: The text query used for finding similar passages.
        text_metadata_df: A Pandas DataFrame containing the text metadata to search.
        column_name: The column name in the text_metadata_df containing the text embeddings or text itself.
        top_n: The number of most similar text passages to return.
        embedding_size: The dimensionality of the text embeddings (only used if text embeddings are stored in the column specified by `column_name`).
        chunk_text: Whether to return individual text chunks (True) or the entire page text (False).
        print_citation: Whether to immediately print formatted citations for the matched text passages (True) or just return the dictionary (False).

    Returns:
        A dictionary containing information about the top N most similar text passages, including cosine scores, page numbers, chunk numbers (optional), and chunk text or page text (depending on `chunk_text`).

    Raises:
        KeyError: If the specified `column_name` is not present in the `text_metadata_df`.
    """

    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")
    
    # Calculate cosine similarity between query text and metadata text
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row,
            column_name,
            get_user_query_text_embeddings(
                project_id, query, embedding_size=128
            ),
        ),
        axis=1,
    )

    # Get top N cosine scores and their indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched text and their information
    final_text = {}

    for matched_textno, index in enumerate(top_n_indices):
        # Create a sub-dictionary for each matched text
        final_text[matched_textno] = {}

        # Store page number
        final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
            "page_num"
        ]

        # Store cosine score
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

        if chunk_text:
            # Store chunk number
            final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                "chunk_number"
            ]

            # Store chunk text
            final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                index
            ]
        else:
            # Store page text
            final_text[matched_textno]["text"] = text_metadata_df["text"][index]

    # Optionally print citations immediately
    #if print_citation:
    #    print_text_to_text_citation(final_text, chunk_text=chunk_text)

    return final_text

def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts an image embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        image_uri (str): The URI (Uniform Resource Identifier) of the image to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the image embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    # image = Image.load_from_file(image_uri)
    image = vision_model_Image.load_from_file(image_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image, contextual_text=text
    )  # 128, 256, 512, 1408
    image_embedding = embeddings.image_embedding

    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding

def get_image_for_gemini_ppts(
    image_bytes: bytes,
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> str:

    # Create the image file name
    image_name = f"{image_save_dir}/{file_name}_image_{page_num}_{image_no}.jpeg"

    # Create the image save directory if it doesn't exist
    os.makedirs(image_save_dir, exist_ok=True)

    # Save the image bytes to a file
    with open(image_name, 'wb') as f:
        f.write(image_bytes)

    # Load the saved image as a Gemini Image Object
    image_for_gemini = Image.load_from_file(image_name)

    return image_name 

def get_image_for_gemini(
    doc: fitz.Document,
    image: tuple,
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> Tuple[Image, str]:
    """
    Extracts an image from a PDF document, converts it to JPEG format, saves it to a specified directory,
    and loads it as a PIL Image Object.

    Parameters:
    - doc (fitz.Document): The PDF document from which the image is extracted.
    - image (tuple): A tuple containing image information.
    - image_no (int): The image number for naming purposes.
    - image_save_dir (str): The directory where the image will be saved.
    - file_name (str): The base name for the image file.
    - page_num (int): The page number from which the image is extracted.

    Returns:
    - Tuple[Image.Image, str]: A tuple containing the Gemini Image object and the image filename.
    """

    # Extract the image from the document
    xref = image[0]
    pix = fitz.Pixmap(doc, xref)

    # Convert the image to JPEG format
    pix.tobytes("jpeg")

    # Create the image file name
    image_name = f"{image_save_dir}/{file_name}_image_{page_num}_{image_no}_{xref}.jpeg"

    # Create the image save directory if it doesn't exist
    os.makedirs(image_save_dir, exist_ok=True)

    # Save the image to the specified location
    pix.save(image_name)

    # Load the saved image as a Gemini Image Object
    image_for_gemini = Image.load_from_file(image_name)

    return image_for_gemini, image_name

def get_user_query_image_embeddings(
    image_query_path: str, embedding_size: int
) -> np.ndarray:
    """
    Extracts image embeddings for the user query image using a multimodal embedding model.

    Args:
        image_query_path: The path to the user query image.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query image embedding.
    """

    return get_image_embedding_from_multimodal_embedding_model(
        image_uri=image_query_path, embedding_size=embedding_size
    )

def get_similar_image_from_query(
    text_metadata_df: pd.DataFrame,
    image_metadata_df: pd.DataFrame,
    query: str = "",
    image_query_path: str = "",
    column_name: str = "",
    image_emb: bool = True,
    top_n: int = 3,
    embedding_size: int = 128,
    project_id: str = "",
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar images from a metadata DataFrame based on a text query or an image query.

    Args:
        text_metadata_df: A Pandas DataFrame containing text metadata associated with the images.
        image_metadata_df: A Pandas DataFrame containing image metadata (paths, descriptions, etc.).
        query: The text query used for finding similar images (if image_emb is False).
        image_query_path: The path to the image used for finding similar images (if image_emb is True).
        column_name: The column name in the image_metadata_df containing the image embeddings or captions.
        image_emb: Whether to use image embeddings (True) or text captions (False) for comparisons.
        top_n: The number of most similar images to return.
        embedding_size: The dimensionality of the image embeddings (only used if image_emb is True).

    Returns:
        A dictionary containing information about the top N most similar images, including cosine scores, image objects, paths, page numbers, text excerpts, and descriptions.
    """
    # Check if image embedding is used
    if image_emb:
        # Calculate cosine similarity between query image and metadata images
        user_query_image_embedding = get_user_query_image_embeddings(
            image_query_path, embedding_size
        )
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_image_embedding),
            axis=1,
        )
    else:
        # Calculate cosine similarity between query text and metadata image captions
        user_query_text_embedding = get_user_query_text_embeddings(
                project_id, query, embedding_size=128
            )
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_text_embedding),
            axis=1,
        )

    # Remove same image comparison score when user image is matched exactly with metadata image
    cosine_scores = cosine_scores[cosine_scores < 1.0]

    # Get top N cosine scores and their indices
    top_n_cosine_scores = cosine_scores.nlargest(top_n).index.tolist()
    top_n_cosine_values = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched images and their information
    final_images: Dict[int, Dict[str, Any]] = {}

    for matched_imageno, indexvalue in enumerate(top_n_cosine_scores):
        # Create a sub-dictionary for each matched image
        final_images[matched_imageno] = {}

        # Store cosine score
        final_images[matched_imageno]["cosine_score"] = top_n_cosine_values[
            matched_imageno
        ]

        # Load image from file
        #final_images[matched_imageno]["image_object"] = Image.load_from_file(
        #    image_metadata_df.iloc[indexvalue]["img_path"]
        #)
        final_images[matched_imageno]["image_object"] = Image.load_from_file(image_query_path)
        #    image_metadata_df.iloc[indexvalue]["img_path"]
        #)

        # Add file name
        final_images[matched_imageno]["file_name"] = image_metadata_df.iloc[indexvalue][
            "file_name"
        ]

        # Store image path
        #final_images[matched_imageno]["img_path"] = image_metadata_df.iloc[indexvalue][
        #    "img_path"
        #]

        # Store page number
        final_images[matched_imageno]["page_num"] = image_metadata_df.iloc[indexvalue][
            "page_num"
        ]

        final_images[matched_imageno]["page_text"] = np.unique(
            text_metadata_df[
                (
                    text_metadata_df["page_num"].isin(
                        [final_images[matched_imageno]["page_num"]]
                    )
                )
                & (
                    text_metadata_df["file_name"].isin(
                        [final_images[matched_imageno]["file_name"]]
                    )
                )
            ]["text"].values
        )

        # Store image description
        #final_images[matched_imageno]["image_description"] = image_metadata_df.iloc[
        #    indexvalue
        #]["img_desc"]
        final_images[matched_imageno]["image_description"] = "Azure Cognitive Services"

    return final_images
