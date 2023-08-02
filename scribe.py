import openai
import os
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from soundfile import SoundFile


in_dir = "in"
out_dir = "out"
# Removed the filename variable
video = None
for file in os.listdir(in_dir):
    print(file)
    if file.endswith((".mp4", ".mkv")):
        print(file)
    video = AudioFileClip(os.path.join(in_dir, file))
    audio_file_name = f"{file.rsplit('.', 1)[0]}.mp3"
    print(audio_file_name)
    video.write_audiofile(os.path.join(out_dir, audio_file_name))
audio_file_name = None
if audio_file_name is not None:
    if os.path.getsize(audio_file_name) > 26214400:
        audio = AudioSegment.from_mp3(audio_file_name)
        audio = audio.set_frame_rate(16000)
        audio.export(audio_file_name, format="mp3")


def transcribe_audio(audio_file_name):
    try:
        with open(audio_file_name, 'rb') as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
        return transcription['text']
    except FileNotFoundError:
        print(f"File {audio_file_name} not found.")
        return None

def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }


def key_points_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def action_item_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def sentiment_analysis(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

# Removed the save_as_file function
            
    

transcription = None
transcription = transcribe_audio(audio_file_name)
minutes = None
if transcription:
    minutes = meeting_minutes(transcription)
    print(minutes)
else:
    print("No transcription available.")

