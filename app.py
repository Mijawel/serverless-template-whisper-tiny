import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("medium")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    end_of_previous_chunk = model_inputs.get('end_of_previous_chunk', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    audio = whisper.load_audio('input.mp3')
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # decode the audio
    options = whisper.DecodingOptions(prefix=end_of_previous_chunk)
    result = whisper.decode(model, mel, options)
    # {"text":result["text"]} TypeError: 'DecodingResult' object is not subscriptable [2022-10-09 04:38:06 +0000]
    output = result.text
    os.remove("input.mp3")
    
    # Return the results as a dictionary
    return output
