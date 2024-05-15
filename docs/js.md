# **Integrating JavaScript and React for the Frontend of Emotion Draw**

For the frontend of Emotion Draw, we leveraged JavaScript and React to create a dynamic and interactive user experience. By combining the power of these technologies with the Chakra UI component library, we crafted a visually appealing and intuitive interface for generating and exploring emotion-based images.

[Let's Go to Our Website! :fontawesome-solid-wand-magic-sparkles:](https://condor-super-halibut.ngrok-free.app/){ .md-button }

## **Key Features Implemented:**

### 1. **Chakra UI Integration:**
   We utilized Chakra UI, a flexible and accessible component library for React, to streamline the development of our frontend. This allowed us to easily implement various UI elements such as headings, containers, buttons, inputs, modals, tooltips, and progress indicators with consistent styling and functionality.

### 2. **State Management with React Hooks:**
   React hooks, including `useState`, `useRef`, and `useDisclosure`, were employed for efficient state management within our application. These hooks enabled us to manage dynamic data, facilitating seamless interaction between components.

### 3. **Asynchronous Data Fetching with Axios:**
   We utilized the Axios library for making asynchronous HTTP requests to our backend server. This allowed us to fetch image data and associated emotions based on user-provided prompts, enabling real-time generation of emotion-based images.

### 4. **Modal and Tooltip Components:**
   We implemented modal and tooltip components using Chakra UI's `Modal` and `Tooltip` components, enhancing user interaction and providing additional context and information. These components offer intuitive ways to display detailed information, instructions, and supplementary content without cluttering the main interface.

### 5. **Event Handling and User Interaction:**
   React event handling mechanisms were employed to manage user interactions such as input submission, image selection, and modal closure. Additionally, we utilized event listeners to trigger image downloads and handle keyboard events for enhanced accessibility and user convenience.

## **Run the JS React Application**

The following command will start the backend and frontend servers simultaneously and open the frontend application and the FastAPI docker in the default web browser.

```shell
$ python run.py
```

Or, alternatively, you can start only the frontend server.

```shell
$ cd Emotion_Draw/client
$ npm start
```

