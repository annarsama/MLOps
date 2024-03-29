# ✨ MLOps Project ✨

This is the final project for a MLOps 💻 lesson from Master 2 SISE (_Université Lumière Lyon 2_) headed by [Fanilo ANDRIANASOLO](https://github.com/andfanilo). The aim of this project was to build a full-stack Dockerized Machine Learning app 📈 with **Streamlit**, a Python framework to build apps easily and quickly.


This app provides an UI to do predictions on a pretrained ML model using Archipelago penguins training dataset 🐧 and a logistic regression.

![image](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.nTBp_OFIa2-7S0dY9-oLMgHaEU%26pid%3DApi&f=1&ipt=1529d0c5a6f93710b5fd3eccd78a075497ecc725ee9ce5a154983629845d6d3c&ipo=images)

## App Folder 📊

The app folder is made up of 2 subfolders:

- *client:* a folder that contains images (`images`), a Python file (`app.py`), a `Dockerfile`, and a requirements file (`.txt`) for installing dependencies.
- *server:* a Python file (`app.py`) for the app, a `Dockerfile`, a pre-trained model (`model.pkl`), a requirements file (`.txt`) for installing dependencies, and a Python file (`train.py`) for the training of the ML model.

The app folder also contains a `docker-compose.yml` file.

## Usage 📍

This app enables you to predict penguin species (i.e., Adelie, Gentoo, Chinstrap) based on certain characteristics (more information on the app's home page).

To launch the app, you need to download the app folder and to launch Docker Desktop first. Then, simply run the following command lines:

```bash
$ cd /path/to/folder/app
$ docker compose up --build
```

## Authors ✏️

Annabelle NARSAMA
