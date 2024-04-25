import subprocess
import webbrowser
import os
from threading import Thread

def start_fastapi():
    subprocess.run(["uvicorn", "Emotion_Draw.api.api:app", "--reload"], shell=True)

def start_frontend():
    os.chdir("Emotion_Draw/client")
    # Start the frontend using npm
    subprocess.run(["npm", "start"], shell=True)

def open_frontend_docs():
    webbrowser.open('http://localhost:3000') 
    webbrowser.open('http://127.0.0.1:8000/docs#/')  

name = "__main__"

if name == "__main__":
    backend_thread = Thread(target=start_fastapi)
    backend_thread.start()

    frontend_thread = Thread(target=start_frontend)
    frontend_thread.start()

    open_frontend_docs()