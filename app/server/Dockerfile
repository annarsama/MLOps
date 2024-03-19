FROM python:3.11-slim

COPY requirements.txt /mlops-td/requirements.txt

WORKDIR /mlops-td 
RUN pip install -r requirements.txt

COPY model.pkl model.pkl
COPY app.py app.py

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app:app"]
# invite de commande :
# uvicorn pour lancer serveur http
# reload pour recharger serveur quand changement du code source
# host pour accepter toutes les connexions
# 0.0.0.0 pour les connexions locales
# app:app pour le fichier principal .py et l'objet instancié "app = FastAPI()"