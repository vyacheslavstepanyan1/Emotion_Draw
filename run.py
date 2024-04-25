"""
Run Script

This script starts the FastAPI application using uvicorn and opens the documentation in a web browser.

Note: Ensure that the uvicorn package is installed before running this script.
"""

import subprocess
import webbrowser
from Emotion_Draw.api import api


def start_fastapi():
    subprocess.run(["uvicorn", "Emotion_Draw.api.api:app", "--reload"])
    webbrowser.open('http://127.0.0.1:8000/docs#/')


name = "__main__"

if name == "__main__":
    start_fastapi()