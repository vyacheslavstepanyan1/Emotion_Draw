# [Emotion Draw](https://vyacheslavstepanyan1.github.io/Emotion_Draw/user_guide/) 🎨

## **Introduction 👋**

"Emotion Draw" combines state-of-the-art technology, utilizing BERT-based models for natural language processing (NLP) to predict emotions from user input sentences. 🤖💬 In tandem, it leverages Stable Diffusion, a powerful deep learning text-to-image model, to generate expressive images corresponding to the predicted emotions. 🎨✨ This fusion of advanced NLP and image generation techniques enables "Emotion Draw" to provide users with a seamless and immersive experience, bridging the gap between textual and visual expression. 🌟🖼️ Through the integration of these cutting-edge AI capabilities, the platform empowers users to explore and communicate their emotions in entirely new and creative ways. 🚀🎭


## **Table of Contents 🤓**

- [**Dataset**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/data/)
    - [Emotions Dataset for NLP](https://vyacheslavstepanyan1.github.io/Emotion_Draw/data/#emotions-dataset-for-nlp)
    - [Preprocessing](https://vyacheslavstepanyan1.github.io/Emotion_Draw/data/#preprocessing)
- [**🎨 Emotion Draw with ALBERT: The Mighty Mite! 🤖**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/bert/)
    - [BERT: The Big, the Bold, and the Brainy - Why We Gave It a Pass! 🫣](https://vyacheslavstepanyan1.github.io/Emotion_Draw/bert/#bert-the-big-the-bold-and-the-brainy-why-we-gave-it-a-pass)
    - [ALBERT: The Chosen One! 🚀](https://vyacheslavstepanyan1.github.io/Emotion_Draw/bert/#albert-the-chosen-one)
    - [How Do the Others Perform?](https://vyacheslavstepanyan1.github.io/Emotion_Draw/bert/#how-do-the-others-perform)
- [**Multiclass Classification Trainer**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/model/)
    - [Initialization](https://vyacheslavstepanyan1.github.io/Emotion_Draw/model/#initialization-init)
    - [Functions](https://vyacheslavstepanyan1.github.io/Emotion_Draw/model/#functions)
- [**Step-by-Step: Fine-Tuning Bert and Friends**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/)
    - [Import Packages](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#import-packages)
    - [Choose a Model](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#choose-a-model)
    - [Specify the Parameters](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#specify-the-parameters)
    - [Initialize the Class](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#initialize-the-class)
    - [Load Data](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#load-data)
    - [Training](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#training)
    - [Evaluate on the Test Set](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#evaluate-on-the-test-set)
    - [Inference for a Single Example](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#inference-for-a-single-example)
    - [Display Confusion Matrices](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#display-confusion-matrices)
    - [TensorBoard](https://vyacheslavstepanyan1.github.io/Emotion_Draw/step_by_step/#tensorboard)
- [**The Artist: Stable Diffusion v2-1 👨🏻‍🎨🎨**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/diffusion/)
    - [Introduction to Stable Diffusion Models](https://vyacheslavstepanyan1.github.io/Emotion_Draw/diffusion/#introduction-to-stable-diffusion-models)
    - [Overview of Stable Diffusion v2-1 Model](https://vyacheslavstepanyan1.github.io/Emotion_Draw/diffusion/#overview-of-stable-diffusion-v2-1-model)
    - [Our Mission: Prompt Engineering 💬](https://vyacheslavstepanyan1.github.io/Emotion_Draw/diffusion/#our-mission-prompt-engineering)
- [**FastAPI Integration**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/fast_api/)
    - [Functionality](https://vyacheslavstepanyan1.github.io/Emotion_Draw/fast_api/#functionality)
    - [Usage](https://vyacheslavstepanyan1.github.io/Emotion_Draw/fast_api/#usage)
    - [Run the FastAPI Application](https://vyacheslavstepanyan1.github.io/Emotion_Draw/fast_api/#run-the-fastapi-application)
    - [Example](https://vyacheslavstepanyan1.github.io/Emotion_Draw/fast_api/#example)
- [**Integrating JavaScript and React for the Frontend of Emotion Draw**](https://vyacheslavstepanyan1.github.io/Emotion_Draw/js/)
    - [Key Features Implemented](https://vyacheslavstepanyan1.github.io/Emotion_Draw/js/#key-features-implemented)
    - [Run the JS React Application](https://vyacheslavstepanyan1.github.io/Emotion_Draw/js/#run-the-js-react-application)


## **Troubleshooting 🎯**

If you encounter any issues, reach out to our team.

## **Contributing 🤝**

We welcome contributions! Feel free to submit bug reports, feature requests, or contribute to the codebase on our [GitHub repository](https://github.com/vyacheslavstepanyan1/Emotion_Draw).

## **Contact Information 📞**

For further assistance or inquiries, contact our team at 

* [anahit_baghdasaryan2@edu.aua.am](mailto:anahit_baghdasaryan3@edu.aua.am)

* [vyacheslav_stepanyan@edu.aua.am](mailto:vyacheslav_stepanyan@edu.aua.am)

## **The Project Structure**

```

├── Emotion_Draw                 # Root directory of the project
│   ├── __init__.py              # Initialization file 
│   ├── api                      # Directory for the API-related code
│   │   ├── __init__.py          # Initialization file 
│   │   └── api.py               # Main API implementation file
│   ├── bert_part                # Directory for BERT-related components
│   │   ├── __init__.py          # Initialization file 
│   │   ├── data                 # Directory for data storage and management
│   │   │   ├── processed        # Processed data files
│   │   │   │   ├── test_data.csv    # Processed test data in CSV format
│   │   │   │   ├── train_data.csv   # Processed training data in CSV format
│   │   │   │   └── val_data.csv     # Processed validation data in CSV format
│   │   │   └── raw              # Raw data files
│   │   │       ├── test.txt     # Raw test data
│   │   │       ├── train.txt    # Raw training data
│   │   │       └── val.txt      # Raw validation data
│   │   ├── inference            # Directory for inference-related code
│   │   │   ├── __init__.py      # Initialization file 
│   │   │   └── inference.py     # Main inference implementation file
│   │   ├── model                # Directory for model-related code
│   │   │   ├── Multiclass_BERT.py   # BERT model fine-tuning implementation for multiclass classification task
│   │   │   └── __init__.py      # Initialization file 
│   │   ├── models_trained       # Directory for storing trained models' checkpoints
│   │   │   └── info.txt         # Information about the directory
│   │   ├── notebooks            # Jupyter notebooks for experiments and analysis
│   │   │   ├── BERT-based_Sequence_Classification.ipynb   # Notebook for BERT-based sequence classification 
│   │   │   └── data_creation.ipynb                        # Notebook for data creation and preprocessing
│   │   ├── reports              # Directory for reports
│   │   │   └── figures          # Figures and visualizations for reports
│   │   │       └── info.txt     # Information about the directory
│   │   └── runs                 # Directory for storing run information and logs
│   │       └── info.txt         # Information about the directory
│   └── client                   # Directory for the client-side application
│       ├── README.md            # Readme file for the client application
│       ├── package-lock.json    # Dependency lock file for npm
│       ├── package.json         # Dependency configuration file for npm
│       ├── public               # Public assets for the client application
│       │   ├── favicon.ico      # Favicon for the client application
│       │   ├── index.html       # Main HTML file for the client application
│       │   ├── logo128.png      # 128x128 logo image
│       │   ├── logo512.png      # 512x512 logo image
│       │   ├── manifest.json    # Web app manifest file
│       │   └── robots.txt       # Robots.txt file for web crawlers
│       └── src                  # Source code for the client application
│           ├── App.js           # Main React component
│           ├── assets           # Assets used in the client application
│           │   ├── bert_monster.png      # BERT monster image
│           │   ├── difus_monster.png     # Diffusion monster image
│           │   └── videoplayback.webm    # Video file
│           ├── index.css        # CSS file for styling
│           └── index.js         # Entry point JavaScript file for the client application
├── LICENSE                      # License file for the project
├── README.md                    # Readme file for the project
├── docs                         # Documentation files
│   ├── bert.md                  # Documentation for BERT-related components
│   ├── data.md                  # Documentation for dataset components
│   ├── diffusion.md             # Documentation for stable diffusion components
│   ├── fast_api.md              # Documentation for FastAPI components
│   ├── img                      # Images used in documentation
│   │   ├── albert.jpeg          # Image of ALBERT model
│   │   ├── bert.webp            # Image of BERT model
│   │   └── diffusion_outputs.png # Image of the stable diffusion model outputs
│   ├── index.md                 # Main documentation index
│   ├── js.md                    # Documentation for JavaScript components
│   ├── model.md                 # Documentation for the Multiclass Classification Trainer component
│   ├── step_by_step.md          # Step-by-step guide for fine-tuning BERT-based models
│   └── user_guide.md            # User guide for the project
├── mkdocs.yml                   # Configuration file for MkDocs documentation generator
├── package-lock.json            # Dependency lock file for npm (root project)
├── requirements.txt             # Python dependencies for the project
├── run.py                       # Main script to run FastAPI Docker
└── setup.py                     # Setup script for packaging the project

```
