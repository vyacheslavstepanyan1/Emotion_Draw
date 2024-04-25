"""
This Python script is designed to facilitate the development and deployment of a web application named Emotion_Draw. 
It includes functions to start the backend server using FastAPI and the frontend application using npm. 
Additionally, it provides a function to open the documentation and the frontend application in the default web browser.

Usage:
- Run the script to start the backend and frontend servers simultaneously and open the frontend application and the FastAPI documentation in the default web browser.
"""
import subprocess
import webbrowser
import os
from threading import Thread

def start_fastapi():
    """
    Start the FastAPI backend server using uvicorn.
    """
    subprocess.run(["uvicorn", "Emotion_Draw.api.api:app", "--reload"], shell=True)

def start_frontend():
    """
    Start the frontend application using npm.
    """
    os.chdir("Emotion_Draw/client")
    subprocess.run(["npm", "start"], shell=True)

def open_frontend_docs():
    """
    Open the documentation and the frontend application in the default web browser.
    """
    webbrowser.open('http://localhost:3000')  # Open the frontend application
    webbrowser.open('http://127.0.0.1:8000/docs#/')  # Open the FastAPI documentation (docker)

name = "__main__"

if name == "__main__":
    # Start the backend and frontend servers in separate threads
    backend_thread = Thread(target=start_fastapi)
    backend_thread.start()

    frontend_thread = Thread(target=start_frontend)
    frontend_thread.start()

    # Open the frontend application and the FastAPI documentation in the default web browser
    open_frontend_docs()
