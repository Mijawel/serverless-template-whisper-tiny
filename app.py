import torch
import whisper
import os
import base64
from io import BytesIO

# Comment to force rebuild
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large")

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
    options = whisper.DecodingOptions(prefix=end_of_previous_chunk,beam_size=5)
    result = whisper.decode(model, mel, options)
    # {"text":result["text"]} TypeError: 'DecodingResult' object is not subscriptable [2022-10-09 04:38:06 +0000]
    output = result.text
    os.remove("input.mp3")

    # check that the resulting text does not contain too many repeated words
    # if yes, we decode the audio again without a prefix
    words = output.split()
    if (len(set(words)) < len(words) / 2) and len(words) > 7:
        result = whisper.decode(model, mel, whisper.DecodingOptions(beam_size=5))
        output = result.text
        # We then remove the number of words that were in thr prefix from the start of the output
        if end_of_previous_chunk:
            words = end_of_previous_chunk.split()
            output = " ".join(output.split()[len(words):])
    # If the output contains no punctuation and is more than 10 words long, we assume that the model has switched
    # to non-punctuation mode and we add some punctuation to the start of the end_of_previous_chunk to trigger it to switch back
    elif not any(char in output for char in ".?!,") and len(words) > 10:
        end_of_previous_chunk = ", " + end_of_previous_chunk
        result = whisper.decode(model, mel, whisper.DecodingOptions(prefix=end_of_previous_chunk, beam_size=5))
        output = result.text
    
    
    # Return the results as a dictionary
    return output
