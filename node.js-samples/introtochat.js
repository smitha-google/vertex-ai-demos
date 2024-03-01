const {VertexAI} = require('@google-cloud/vertexai');

/**
 * TODO(developer): Update these variables before running the sample.
 */
// Initialize Vertex with your Cloud project and location
const vertex_ai = new VertexAI({project: 'smithaargolisinternal', location: 'northamerica-northeast1'});
const model = 'gemini-pro';

// Instantiate the models
const generativeModel = vertex_ai.preview.getGenerativeModel({
    model: model,
    generation_config: {
      "max_output_tokens": 2048,
      "temperature": 0.9,
      "top_p": 1
  },
  });
async function startChat() {
    console.log('hi')

    const chat = generativeModel.startChat({context:"You are Lola, a customer service chatbot for Google. You only answer customer questions about Google and its products."});
    const chatInput1 = 'Where can I learn more about Gemini Model offered by Google?';

    console.log(`User: ${chatInput1}`);
    const result1 = await chat.sendMessageStream(chatInput1);
    for await (const item of result1.stream) {
      console.log(item.candidates[0].content.parts[0].text);
    }
}

startChat();