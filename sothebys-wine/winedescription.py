import pandas as pd
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate(prompt):
  vertexai.init(project="smithaargolisinternal", location="us-central1")
  model = GenerativeModel("gemini-1.0-pro-001")
  responses = model.generate_content(
      prompt,
      generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.5,
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

def process():
   # Read the CSV file into a DataFrame
    rawdf = pd.read_csv('./data_export.csv')
    #rawdf = rawdf.dropna()

    cleandf = buildcleandf(rawdf)

    # Print the DataFrame
    for i in cleandf.iloc:
       workingdf = i
       buildprompt(workingdf)

def buildcleandf(rawdf) -> pd.DataFrame:
   #Build a CleanDataframe with needed columns - 
   cleandf = pd.DataFrame(columns=['Title', 'Producer', 'Vintage', 'SubRegion','Region','Country',
                                'Color', 'WineType', 'Grape', 'Classification']) 
   cleandf["Title"] = rawdf["Title"]
   cleandf["Producer"] = rawdf["Metafield: details.producer [single_line_text_field]"]
   cleandf["Vintage"] = rawdf["Metafield: details.vintagenv [single_line_text_field]"]
   cleandf["SubRegion"] = rawdf["Metafield: details.subregion [single_line_text_field]"]
   cleandf["Region"] = rawdf["Metafield: details.region1 [single_line_text_field]"]
   cleandf["Country"] = rawdf["Metafield: details.country [single_line_text_field]"]
   cleandf["Color"] = rawdf["Metafield: details.color [single_line_text_field]"]
   cleandf["WineType"] = rawdf["Type"]
   cleandf["Grape"] = rawdf["Metafield: details.grape [list.single_line_text_field]"]
   cleandf["Classification"] = rawdf["Metafield: details.classification [single_line_text_field]"]
   
   cleandf = cleandf.dropna()
 
   return cleandf

def buildprompt(workingdf):
   #Build the prompt-
    instructions = "You are a wine expert. You are given the labels for a bottle of wine and \
    your role is to generate descriptions. The voice has to be authoritative or crisp. \
    Do not use flowery or descriptive language."

    prompt = f"""Write a product description in no more than 
    5 sentences for a high-end wine bottle with the following 
    specifications: 

    Winery: {workingdf["Title"]}
    Producer: {workingdf["Producer"]}
    Region: {workingdf["Region"]}
    SubRegion: {workingdf["SubRegion"]}
    Country: {workingdf["Country"]}
    Classification: {workingdf["Classification"]}
    Vintage: {workingdf["Vintage"]}
    Varietal Blend: {workingdf["Grape"]}
    Style: {workingdf["Color"]} {workingdf["WineType"]}

    Your task focuses generating descriptions only based on the labels provided as input. 
    ALWAYS start the description with Title and use the region and country where the wine is from in your description. 

    Format these descriptions in a bulluted list with Descriptions, Tasting Notes and Suggested Pairings.  
    
    """

    final_prompt = f""" {prompt}

        {instructions}

        Answer:
        """
    
    responses = generate(final_prompt)
    response_text = ""

    for response in responses:
        #print(response.text, end="")
        response_text = response_text + response.text
    
    print(f"""{workingdf["Title"]}|{workingdf["Producer"]}|{workingdf["Region"]}|{workingdf["SubRegion"]}|{workingdf["Country"]}|{workingdf["Classification"]}|{workingdf["Vintage"]}|{workingdf["Grape"]}|{workingdf["Color"]}|{workingdf["WineType"]}|{response_text}""")

process()
