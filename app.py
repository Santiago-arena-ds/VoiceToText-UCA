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
import whisper
import tempfile
psswd = "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"
from huggingface_hub import login
login(psswd)
from transformers import AutoTokenizer
from transformers import AutoConfig
import torch
import transformers
from transformers import AutoModelForCausalLM
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
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

def get_resumen(texto):
    chat = [
        { "role": "user", "content": f":make a detailed and extended writtens summary of the following: {texto}... in spanish" },

    ]
    return chat

def resumen(texto):
    main_directory = os.getcwd()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #relative_path = path
    #archivo_txt = os.path.join(main_directory, relative_path)
    #print("Absolute file path:", archivo_txt)
    #with open(archivo_txt, "r", encoding="utf-8") as file:
    #    texto = file.read()
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")
    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    config = AutoConfig.from_pretrained(model_name)
    #token_limit = config.max_position_embeddings
    #print("Los token maximos del modelo son:" +" "+ str(token_limit))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ##
    #prompt = texto
    #encoded_prompt = tokenizer.encode(prompt,add_special_tokens=True,truncation=False)
    #token_texto = len(encoded_prompt)
    #print(token_texto)
    #secciones = round(token_texto/token_limit)
    #print("serán {} secciones de texto para el modelo {}".format(secciones,model_name))
    #largo = len(texto)//3
    ##
    #print("el largo es "+ str(largo)+" de "+str(len(texto)))
    #prompt_1= prompt[:len(texto)//1]
    #len(tokenizer.encode(prompt,add_special_tokens=True,truncation=False))
    #len(tokenizer.encode(prompt_1,add_special_tokens=True,truncation=False))
    prompt = tokenizer.apply_chat_template(get_resumen(texto), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=8196)
    #prompt = f"""Instruct: Hacé un resumen cuya longitud sea moderada: '{texto}'.\nOutput:"""
    #model_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    #outputs = model.generate(**model_inputs,max_length=token_limit)
    text = tokenizer.batch_decode(outputs)[0]
    #text=text[text.find('Output:')+9:len(text)-5] #Eliminar <eos> y 'Output'.
    text=respuesta(text)
    return text

def respuesta(texto):
    position_model = texto.find("model")
    position_eos = texto.find('<eos>')
    rta = texto[position_model+len('model'):position_eos] #-len("<eos>")
    return rta
 


#NO ESTÁ ANDANDO AHORA
def hacer_resumen2(texto):          # PASANDO COMO ARGUMENTO EL STRING DEL TEXTO
    
    """
    model_id = "google/gemma-2b-it"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    """
    
    main_directory = os.getcwd()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #relative_path = path
    #archivo_txt = os.path.join(main_directory, relative_path)
    #print("Absolute file path:", archivo_txt)
    #with open(archivo_txt, "r", encoding="utf-8") as file:
    #    texto = file.read()
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype)
    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    config = AutoConfig.from_pretrained(model_name)
    token_limit = config.max_position_embeddings
    #print("Los token maximos del modelo son:" +" "+ str(token_limit))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ##
    prompt = texto
    encoded_prompt = tokenizer.encode(prompt,add_special_tokens=True,truncation=False)
    token_texto = len(encoded_prompt)
    #print(token_texto)
    secciones = round(token_texto/token_limit)
    #print("serán {} secciones de texto para el modelo {}".format(secciones,model_name))
    largo = len(texto)//3
    ##
    #print("el largo es "+ str(largo)+" de "+str(len(texto)))
    prompt_1= prompt[:len(texto)//1]
    len(tokenizer.encode(prompt,add_special_tokens=True,truncation=False))
    len(tokenizer.encode(prompt_1,add_special_tokens=True,truncation=False))
    
    
    prompt = tokenizer.apply_chat_template(get_prompt(texto), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=8196)
    
    
    
    prompt = f"""Instruct: Hacé un resumen cuya longitud sea moderada: '{texto}'.\nOutput:"""
    model_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**model_inputs,max_length=token_limit)
    text = tokenizer.batch_decode(outputs)[0]
    #text=text[text.find('Output:')+9:len(text)-5] #Eliminar <eos> y 'Output'.
    
    res=respuesta(text)
    return res


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
    ["Transcribir",'Resumir', "Punteo"],captions=['Copia lo más preciso posible todo lo dicho en el audio','Resumí lo grabado','Hacé un punteo de lo más importante'])


audio=st.file_uploader('Adjuntá el archivo - Tamaño máximo 200MB')



boton = st.button('Procesar')


if boton == True and audio is not None:
    st.success('Archivo cargado correctamente', icon="✅")
    
    
    
    if accion =='Transcribir':
        ruta_temporal=ruta_a_temporario()
        mensaje_de_prueba = transcribe_rapido(ruta_temporal)
               
    elif accion == 'Resumir':
        ruta_temporal=ruta_a_temporario()  
        transcripcion = transcribe_rapido(ruta_temporal)
        mensaje_de_prueba=resumen(transcripcion)
             
    elif accion == 'Punteo':
        mensaje_de_prueba='punnnteo'
        
    with st.chat_message("assistant"):
        st.write("Hola! Este es el resultado: \n")
        st.write(mensaje_de_prueba)
