# MNIST Classification – Projet Deep Learning

Ce projet consiste à entraîner des modèles de classification d’images sur la base MNIST, et à mettre en production le modèle via une API FastAPI et un frontend Streamlit. Le tout est containerisé avec Docker.

---

## 1. Objectif

L’objectif est de comparer deux architectures de modèles (MLP et CNN), d’explorer l’effet de permutations aléatoires des pixels, de mettre en place une API d’inférence et de proposer une interface web de prédiction de chiffres manuscrits dessinés en temps réel.

---

## 2. Préparation des données

Le dataset MNIST est téléchargé avec `torchvision.datasets.MNIST`. Un prétraitement est appliqué :

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

Dans certains cas, des augmentations sont ajoutées pour rendre le modèle plus robuste aux variations manuscrites :

```python
transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), shear=15)
transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
```

---

## 3. Architecture des modèles

### 3.1. MLP (Multilayer Perceptron)

Le modèle MLP est défini avec 3 couches linéaires et des activations ReLU intermédiaires. Il est entraîné sur les pixels aplatis (28×28).

### 3.2. CNN (Convolutional Neural Network)

Un modèle convolutionnel simple est défini dans `convnet.py`, avec 2 couches `Conv2d`, des activations ReLU, du `MaxPool2d` et une couche fully connected finale.

---

## 4. Entraînement du modèle

L’entraînement est réalisé dans `train.py`. La fonction `train()` applique :

- mise en mode `train()`
- calcul de la loss `F.cross_entropy`
- propagation arrière (`loss.backward()`)
- update avec `optimizer.step()`

Une permutation de pixels peut être appliquée avec `perm = torch.randperm(784)`, ce qui perturbe fortement les modèles, sauf pour le MLP.

La fonction `test()` permet de mesurer la `test_loss` et l’`accuracy`.

### Remarques :

- Le CNN est bien plus performant qu’un MLP sur des images structurées.
- En cas de permutation aléatoire, les CNN perdent leur avantage (perdent la spatialité), contrairement aux MLP.
- Une augmentation des données améliore les performances de généralisation sur des chiffres dessinés à la main.

---

## 5. Sauvegarde du modèle

Le modèle est sauvegardé dans `model/mnist-0.0.1.pt` avec :

```python
torch.save(model.state_dict(), "model/mnist-0.0.1.pt")
```

---

## 6. API avec FastAPI

Une API est exposée dans `main.py` via FastAPI :

- `POST /api/v1/predict` prend en entrée un fichier image PNG
- Convertit l’image en tenseur normalisé 28×28
- Prédit la classe avec le modèle CNN pré-entraîné

---

## 7. Frontend avec Streamlit

L’interface graphique Streamlit permet :

- De dessiner un chiffre dans une zone de dessin (canvas)
- De visualiser l’image 28×28 envoyée à l’API
- D’afficher la prédiction retournée

Traitement appliqué sur l’image dessinée :

- Inversion des couleurs (dessin blanc sur fond noir)
- Resize à 28×28
- Sauvegarde PNG temporaire pour appel API

---

## 8. Dockerisation

Deux Dockerfiles sont créés :

- `Dockerfile.api` pour le backend FastAPI
- `Dockerfile.front` pour le frontend Streamlit

Un `docker-compose.yml` permet de lancer le projet complet avec :

```bash
docker compose up --build
```

---

## 9. Tests

Des tests d’image sont faits avec :

- Des images extraites de MNIST (via script matplotlib)
- Des chiffres dessinés manuellement avec la souris

On observe que :

- Les prédictions sur des images dessinées à la main sont correctes uniquement si l'entraînement a été fait avec des augmentations réalistes
- Le CNN bien entraîné atteint >98% de précision sur MNIST et fonctionne bien sur les dessins

---

## 10. Conclusion

Ce projet illustre l’intérêt des réseaux de neurones convolutionnels pour la classification d’images structurées. Il montre aussi comment entraîner, sauvegarder et déployer un modèle en production à travers une API REST et une interface web simple.

Points clés :

- Le MLP est plus robuste aux permutations aléatoires
- Le CNN tire profit de la structure spatiale de l’image
- La mise en production nécessite une standardisation de l’entrée
- L’ajout d’augmentations (affine, perspective) est crucial pour généraliser

---

## Lancement

Une fois Docker installé :

```bash
docker compose up --build
```

Puis ouvrir [http://localhost:8501](http://localhost:8501)