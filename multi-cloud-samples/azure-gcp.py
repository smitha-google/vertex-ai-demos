import openai
import faiss
import numpy as np

openai.api_key = "YOUR_OPENAI_API_KEY"

# Sample data (replace with your own)
data = [
    "The Earth is a planet.", 
    "The sky is blue.",
    "Coding can be fun.",
    "AI is a powerful technology."
]

# Function to generate embeddings
def create_embedding(text):
    embedding = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")['data'][0]['embedding']
    return embedding

# Create embeddings for your data
embeddings = np.array([create_embedding(text) for text in data])

# Build a Faiss index
dimension = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Function to handle a user's question
def answer_question(question):
    question_embedding = create_embedding(question)
    distances, indices = index.search(np.array([question_embedding]), k=3)  # Find top 3 similar

    retrieved_text = [data[index] for index in indices[0]]

    prompt = "Instructions: Use the following information to answer the question. Remember to be helpful and informative.\n\nRetrieved Information: {}\n\n Question: {}\n\nAnswer:".format("\n".join(retrieved_text), question)

    response = openai.Completion.create(
        engine="text-davinci-003",  # Or another Gemini model
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

# Example usage
user_question = "What color is the sky?"
answer = answer_question(user_question)
print(answer) 