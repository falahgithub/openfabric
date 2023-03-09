import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer          # importing pretrained model and tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")                  # initializing tokenizer
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)    # initializing model


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # Extracting relevant information from the user's input

    # Let's extract name
    name_e = extract_name(input_text)             # function defined to extract name

    # Updating configuration
    if name_e is not None:
        configuration.name = name_e

    # Return the updated configuration
    return configuration



############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        input_tf = tokenizer.encode(text, return_tensors="tf")
        beam_output = model.generate(input_tf, max_length=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_gen = tokenizer.decode(beam_output[0], skil_special_tokens=True, clean_up_tokenization_spaces=True)

        response = output_gen.split("\n\n")[1]
        output.append(response)

    return SimpleText(dict(text=output))
