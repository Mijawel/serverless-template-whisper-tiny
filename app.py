import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global sanity_check_model
    
    model = whisper.load_model("large")
    sanity_check_model = whisper.load_model("base")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global sanity_check_model

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
    output = result.text

    # do the sanity check
    sanity_options = whisper.DecodingOptions()
    sanity_result = whisper.decode(sanity_check_model, mel, sanity_options)
    sanity_output = sanity_result.text

    retry_needed = False
    
    # check that the lengths are approximately the same
    if (abs(len(sanity_output) - (len(output)+len(end_of_previous_chunk))) > 10):
        retry_needed = True

    # check that the sets of words in the two outputs are approximately the same
    sanity_set = set(sanity_output.split())
    regular_set = set(output.split())
    regular_set.update(set(end_of_previous_chunk.split()))

    if (len(sanity_set.symmetric_difference(regular_set)) > 10):
        retry_needed = True

    if (retry_needed):
        options = whisper.DecodingOptions(beam_size=5)
        result = whisper.decode(model, mel, options)
        output = result.text
    
    os.remove("input.mp3")
    
    return {'output':output,'retry_needed':retry_needed}
