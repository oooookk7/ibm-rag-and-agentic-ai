import os

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from transformers import pipeline  # For Speech-to-Text

#######------------- LLM Initialization-------------#######

load_dotenv()

model_id = "moonshotai/Kimi-K2-Instruct"
provider = "novita"

# Initialize Hugging Face chat model
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        provider=provider,
        temperature=0.5,
        max_new_tokens=512,
    )
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    fix_prompt = (
        "You are a product terminology assistant.\n"
        "Correct obvious product-name spelling errors and normalize product terms.\n"
        "Keep the original meaning unchanged.\n\n"
        f"Transcript:\n{ascii_transcript}\n\n"
        "Corrected Transcript:"
    )
    return llm.invoke(fix_prompt).content


#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    prompt | llm | StrOutputParser()
)

#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file.", None

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
    )

    # Gradio with type="numpy" returns (sample_rate, waveform).
    sample_rate, waveform = audio_file
    if waveform is None:
        return "Invalid audio input.", None

    # Convert stereo to mono and normalize integer PCM to float32 in [-1, 1].
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        max_val = np.iinfo(waveform.dtype).max
        waveform = waveform.astype(np.float32) / max_val
    else:
        waveform = waveform.astype(np.float32)

    raw_transcript = pipe(
        {"array": waveform, "sampling_rate": sample_rate},
        batch_size=8,
        return_timestamps=True,
    )["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    # Write the result to a file for downloading
    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    # Return the textual result and the file for download
    return result, output_file


#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="numpy", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="0.0.0.0", server_port=5001)
