#!/bin/bash
echo "Lancement du pipeline MNIST (${1:-cnn})..."

source monenv310/bin/activate

# Change le type de modèle directement dans le fichier Python
MODEL_TYPE=${1:-cnn}
sed -i "" "s/MODEL_TYPE = \".*\"/MODEL_TYPE = \"$MODEL_TYPE\"/" main_pipeline.py

# Lance le script
python main_pipeline.py
