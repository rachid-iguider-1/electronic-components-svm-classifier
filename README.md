# 🔍 Classification d'Images de Composants Électroniques — Support Vector Machine (SVM)

Ce projet implémente un pipeline de **Computer Vision** et de **Machine Learning** pour classifier automatiquement des images de composants électroniques (*Résistances* et *Potentiomètres*) en utilisant un modèle **SVM (Support Vector Machine)** optimisé par validation croisée.

---

## 📊 Performances du Modèle

* **Précision globale (Accuracy) :** **90.00 %**
* **Meilleurs hyperparamètres (GridSearch CV) :**
  * `Kernel` : `rbf`
  * `C` : `1`
  * `Gamma` : `auto`
* **Score de F1 par classe :**
  * **Potentiomètre :** $0.90$
  * **Résistance :** $0.90$

### 🧪 Matrice de Confusion (Données de Test : 60 images)

| | Prédit : Potentiometer | Prédit : Resistor |
| :--- | :---: | :---: |
| **Réel : Potentiometer** | **27** | 3 |
| **Réel : Resistor** | 3 | **27** |

---

## 🛠️ Pipeline de Traitement des Données

1. **Prétraitement d'image (OpenCV) :**
   * Chargement en niveaux de gris (`IMREAD_GRAYSCALE`).
   * Redimensionnement uniforme à $64 \times 64$ pixels.
   * Aplatissement des images en vecteurs de $4096$ caractéristiques ($64 \times 64$).
2. **Préparation des données (Scikit-Learn) :**
   * Stratification de l'échantillon : $80\%$ Entraînement ($240$ images) et $20\%$ Test ($60$ images).
   * Normalisation des caractéristiques via `StandardScaler`.
3. **Optimisation & Validation Croisée :**
   * Recherche par grille (`GridSearchCV`) sur $5$ plis (5-Fold Cross-Validation) à travers $12$ combinaisons d'hyperparamètres ($C$, $Kernel$, $Gamma$).
4. **Inférence en Temps Réel :**
   * Fonction d'inférence autonome (`predict_new_image`) permettant de classifier de nouvelles images.

---

## 📁 Structure du Projet

```text
.
├── electronics_dataset/
│   └── train_set/
│       ├── resistor/          # Images de résistances
│       └── potentiometer/     # Images de potentiomètres
├── train_classifier.py        # Script principal (Chargement, entraînement et prédiction)
└── README.md                  # Documentation
