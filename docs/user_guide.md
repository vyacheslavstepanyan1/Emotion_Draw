# **Enotion Draw User Guide**

## **Introduction 👋**

"Emotion Draw" combines state-of-the-art technology, utilizing BERT-based models for natural language processing (NLP) to predict emotions from user input sentences. 🤖💬 In tandem, it leverages Stable Diffusion, a powerful deep learning text-to-image model, to generate expressive images corresponding to the predicted emotions. 🎨✨ This fusion of advanced NLP and image generation techniques enables "Emotion Draw" to provide users with a seamless and immersive experience, bridging the gap between textual and visual expression. 🌟🖼️ Through the integration of these cutting-edge AI capabilities, the platform empowers users to explore and communicate their emotions in entirely new and creative ways. 🚀🎭


## **Table of Contents 🤓**

- [**Dataset**](data.md)
    - [Emotions Dataset for NLP](data.md#emotions-dataset-for-nlp)
    - [Preprocessing](data.md#preprocessing)
- [**🎨 Emotion Draw with ALBERT: The Mighty Mite! 🤖**](bert.md)
    - [BERT: The Big, the Bold, and the Brainy - Why We Gave It a Pass! 🫣](bert.md#bert-the-big-the-bold-and-the-brainy---why-we-gave-it-a-pass-🫣)
    - [ALBERT: The Chosen One! 🚀](bert.md#albert-the-chosen-one-🚀)
    - [How Do the Others Perform?](bert.md#how-do-the-others-perform)
- [**Multiclass Classification Trainer**](model.md)
    - [Initialization](model.md#initialization-init)
    - [Functions](model.md#functions)
- [**Step-by-Step: Fine-Tuning Bert and Friends**](step_by_step.md)
    - [Import Packages](step_by_step.md#import-packages)
    - [Choose a Model](step_by_step.md#choose-a-model)
    - [Specify the Parameters](step_by_step.md#specify-the-parameters)
    - [Initialize the Class](step_by_step.md#initialize-the-class)
    - [Load Data](step_by_step.md#load-data)
    - [Training](step_by_step.md#training)
    - [Evaluate on the Test Set](step_by_step.md#evaluate-on-the-test-set)
    - [Inference for a Single Example](step_by_step.md#inference-for-a-single-example)
    - [Display Confusion Matrices](step_by_step.md#display-confusion-matrices)
    - [TensorBoard](step_by_step.md#tensorboard)
- [**The Artist: Stable Diffusion v2-1 👨🏻‍🎨🎨**](diffusion.md)
    - [Introduction to Stable Diffusion Models](diffusion.md#introduction-to-stable-diffusion-models)
    - [Overview of Stable Diffusion v2-1 Model](diffusion.md#overview-of-stable-diffusion-v2-1-model)
    - [Our Mission: Prompt Engineering 💬](diffusion.md#our-mission-prompt-engineering-💬)
- [**FastAPI Integration**](fast_api.md)
    - [Functionality](fast_api.md#functionality)
    - [Usage](fast_api.md#usage)
    - [Run the FastAPI Application](fast_api.md#run-the-fastapi-application)
    - [Example](fast_api.md#example)
- [**Integrating JavaScript and React for the Frontend of Emotion Draw**](js.md)
    - [Key Features Implemented](js.md#key-features-implemented)
    - [Run the JS React Application](js.md#run-the-js-react-application)


## **Troubleshooting 🎯**

If you encounter any issues, reach out to our team.

## **Contributing 🤝**

We welcome contributions! Feel free to submit bug reports, feature requests, or contribute to the codebase on our [GitHub repository](https://github.com/vyacheslavstepanyan1/Emotion_Draw).

## **Contact Information 📞**

For further assistance or inquiries, contact our support team at 

[anahit_baghdasaryan2@edu.aua.am :fontawesome-solid-paper-plane:](mailto:anahit_baghdasaryan3@edu.aua.am){ .md-button }
[vyacheslav_stepanyan@edu.aua.am :fontawesome-solid-paper-plane:](mailto:vyacheslav_stepanyan@edu.aua.am){ .md-button }
