# Scriber

A simple scribe that uses OpenAIs API Whisper-1 model to encode the speach into text. Afterward it uses GPT-4 to perform an abstract summary, create a list of action items, keynote take always and performs a sentiment analysis. It outputs both the full transcript and the analysis. 

## Install

### Create the environment

run
`python -m venv .venv`

### Activate environment

Linux

run
`source .venv/bin/activate`

Windows

run
`./.venv/Scripts/activate.bat`

### Install requirements

run
`pip install -r requirements.txt`

## Usage

Currently works on *mp3*, *mkv*, and *mp4*. Put the target file in the *in* folder.
Copy the .env.example file and rename it to *.env*. Put your [OpenAI API Key](https://platform.openai.com/account/api-keys) into the .env file on the right side of the *=*.

run
`python scriber.py`

That should output a *minutes.txt* and *transcript.txt* in the out folder and a *FILE_NAME.mp3* file in the in folder if you put in an audio file.

If there are any issues please open a ticket I'll do my best to help.

The OpenAI API requires a credit card on file and charges for its API usage.

Fully open source and licenced under the MIT license.
