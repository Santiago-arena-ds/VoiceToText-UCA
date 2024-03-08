# IMPORTS-------------------------------------------------------------------------------------------
import subprocess

# Define the command to install requirements.txt
command = "pip install -r requirements.txt"

# Execute the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
    print("Requirements installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error installing requirements: {e}")
import streamlit as st
import pandas as pd
import tempfile
psswd = "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"
from huggingface_hub import login
import transformers
login(psswd)
from transformers import AutoTokenizer
from transformers import AutoConfig
import torch

from transformers import AutoModelForCausalLM
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time




def transcribe_rapido(path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=8,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    #fin=time.time()
    #print(f'Elapsed time: {fin - ini}s')
    #ini2=time.time()
    sample = path
    result = pipe(sample, generate_kwargs={"language": "spanish"})
    #fin2=time.time()
    #print(f'Elapsed time: {fin2 - ini2}s')

    texto = result["text"]
    return texto

#----------------------------------------------------



def ruta_a_temporario():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file: #ok
            temp_file.write(audio.read()) #ok
            ruta_temporal = temp_file.name #ok
    return ruta_temporal

#Estética de la página---------------------------------------------------------------------------------
st.cache_data()
st.title('Laboratorio de Ciencia de Datos - UCA')
st.divider()
st.title('Speech to text')
st.text('Aquí podrás adjuntar un archivo de audio y podrás elegir si querés la transcripción,\nun resumen o una itemización.')
#-----------------BOTONERA------------------------




accion = st.radio(
    "Qué deseas hacer?",
    ["Transcribir"],captions=['Copia lo más preciso posible todo lo dicho en el audio','Resumí lo grabado','Hacé un punteo de lo más importante'])


audio=st.file_uploader('Adjuntá el archivo - Tamaño máximo 200MB')



boton = st.button('Procesar')


if boton == True and audio is not None:
    st.success('Archivo cargado correctamente', icon="✅")
    
    
    
    if accion =='Transcribir':
        ruta_temporal=ruta_a_temporario()
        mensaje_de_prueba = transcribe_rapido(ruta_temporal)
               
             
        
    with st.chat_message("assistant"):
        st.write("Hola! Este es el resultado: \n")
        st.write(mensaje_de_prueba)
