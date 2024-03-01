const {VertexAI} = require('@google-cloud/vertexai');

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

async function generateContent() {
  const req = {
    contents: [{role: 'user', parts: [{text: "What is a large language model?"}]}],
  };

  console.log('hi')

  const streamingResp = await generativeModel.generateContentStream(req);

  // Wait for the response stream to complete
  const aggregatedResponse = await streamingResp.response;

  // Select the text from the response
  const fullTextResponse =
    aggregatedResponse.candidates[0].content.parts[0].text;

  console.log(fullTextResponse);

  //for await (const item of streamingResp.stream) {
    //process.stdout.write('stream chunk: ' + item);
  //}

  //process.stdout.write('aggregated response: ' + (await streamingResp.response));
};

generateContent();