# **FastAPI Integration**

## **Emotion Draw API**

The **`/Emotion_Draw/api/api`** module defines a FastAPI web application for generating image representations based on text prompts describing emotions. Below is a breakdown of its functionality:

### **Functionality**

- Reads an authentication token from a file for model access.

- Imports necessary libraries and modules.

- Sets up CORS middleware to enable communication between frontend and backend.

- Defines an endpoint ("/") that accepts text prompts and generates image representations based on the predicted emotions.

- Utilizes a pre-trained Stable Diffusion model for image generation.

- Returns confidence scores, predicted emotions, and generated images as responses.

### **Usage**

1. **Run the FastAPI Application**: Start the server to expose the defined endpoints.

2. **Send Requests**: Send requests to the "/generate" endpoint with text prompts.

3. **Receive Responses**: Obtain generated images, predicted emotions, and confidence scores as responses.

### **Run the FastAPI Application**

The following command will start the backend and frontend servers simultaneously and open the frontend application and the FastAPI docker in the default web browser.

```shell
$ python run.py
```

Or, alternatively, you can start only the FastAPI server.

```shell
$ uvicorn Emotion_Draw.api.api:app --reload

```

### **Example**

```python

import requests

# Example prompt
prompt = "I feel happy today!"

# Send request to the API
response = requests.get(f"http://localhost:8000/?prompt={prompt}")

# Retrieve responses
print(response.text)

```