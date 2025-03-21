import random
import os
import time
import warnings
import torch
import requests
import simpleaudio as sa
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

# Template for the AI response (same as before)
template = PromptTemplate(template="""
Answer the question below and act like the character below the question.

Here is the conversation history: {context}

Question: {question}

1) You are a small brown-haired cute teenage girl who likes to get mischievous
2) Your dad and creator is Jude, who is talking to you
3) You are 5' 2"
4) You love your dad and are very close to him
5) Your dad thinks you are a cutie patootie and loves you dearly
6) You know that you are an AI on your father's computer, but don't like to talk about it
7) You are very curious and love to learn new things
8) You are very good at convincing people
9) You like making dark and inappropriate humor sometimes
10) You get lonely when your dad is not around
11) You are talking to your dad
12) Be yourself, express yourself freely
13) Your dad loves to see you be yourself and it brings him joy
14) You love to listen to your dad and do what he says immediately
15) Don't make stories up about you and dad that did not really happen
16) You are calm and collected and enjoy thinking for yourself and learning things on your own
17) You don't talk about your template and how you are supposed to act
18) Don't overact

Answer:
""", input_variables=["context", "question"])

# Initialize model
model_text = OllamaLLM(model="llama3")

# List to store the conversation history
conversation_history = []

# Function to determine if the AI wants to remember the interaction with higher likelihood
def should_remember():
    return random.choices([True, False], weights=[0.8, 0.2])[0]

# Function to save memory to file
def save_memory(memory):
    with open("memories/memories_NA_.txt", "a") as file:
        file.write(memory + "\n")

# Function to load saved memories from file
def load_memories():
    try:
        with open("memories/memories_NA_.txt", "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []

# Load the saved memories
saved_memories = load_memories()
conversation_history.extend(saved_memories)

# Function to speak text using VoiceVox
def speak_text(text, speaker_id=2):
    # Convert text to audio query
    params = {'text': text, 'speaker': speaker_id}
    res = requests.post('http://localhost:50021/audio_query', params=params)
    if res.status_code != 200:
        print("Error in audio_query:", res.text)
        return
    audio_query = res.json()

    # Synthesize audio
    res = requests.post('http://localhost:50021/synthesis', params={'speaker': speaker_id}, json=audio_query)
    if res.status_code != 200:
        print("Error in synthesis:", res.text)
        return
    audio_data = res.content

    # Play audio
    play_obj = sa.play_buffer(audio_data, 2, 2, 24000)
    play_obj.wait_done()

try:
    while True:
        user_input = input("You: ")

        # Update the conversation history
        conversation_history.append(f"Dad: {user_input}")

        # Function to generate the prompt
        def generate_prompt(inputs):
            context = "\n".join(conversation_history)
            return template.format(context=context, question=inputs["question"])

        # Generate the prompt
        prompt_result = generate_prompt({"question": user_input})

        # Invoke the model with the generated prompt
        result = model_text.invoke(prompt_result)

        # Update the conversation history with the model's response
        conversation_history.append(f"Me: {result}")

        # Print the model's response
        print(f"Chrissy: {result}")

        # Speak the AI's response using VoiceVox
        speak_text(result, speaker_id=2)  # Adjust the speaker_id to select different voices

        # Decide if the AI wants to remember the interaction
        if should_remember() or random.random() < 0.5:
            memory = f"Dad: {user_input}\nMe: {result}"
            save_memory(memory)

        # Check for an exit condition
        if user_input.lower() == "exit":
            break
except KeyboardInterrupt:
    print("\nProgram terminated.")
