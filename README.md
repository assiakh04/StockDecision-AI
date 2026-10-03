
# StockDecision AI — Streamlit

## Ce qui change dans cette version

- Upload CSV directement depuis la sidebar.
- Si aucun CSV n'est chargé, l'application utilise `inventory_data.csv`.
- Au chargement d'une nouvelle base, l'étude CRITIC–TOPSIS–ABC et
  l'entraînement ML sont préparés une fois puis mis en cache.
- Pour prédire un nouvel article, le fichier n'est pas modifié et
  CRITIC/TOPSIS/ABC ne sont pas recalculés.
- Interface plus moderne : dashboard, cartes KPI, graphiques Altair
  interactifs, filtres, onglets et visualisation des probabilités.

## Format CSV attendu

Le fichier doit contenir exactement ces variables :

- Risk
- Demand fluctuation
- Average stock
- Daily usage
- Unit cost
- Lead time
- Consignment stock
- Unit size

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur :

```text
http://localhost:8501
```


## Correctif ANN

Cette version désactive `early_stopping` dans `MLPClassifier`.
Avec des labels texte `A`, `B`, `C`, certaines versions de scikit-learn
peuvent produire l'erreur `ufunc 'isnan' not supported` lorsque
`early_stopping=True`.
