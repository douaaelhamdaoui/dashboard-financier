# Comparaison des Optimiseurs : SGD | Adam | Nadam

Implémentation from scratch en Python des trois optimiseurs principaux du Deep Learning.

## Contenu
- `optimizers_comparison.py` : code complet (MLP + SGD + Adam + Nadam)
- `comparaison.png` : courbes Loss et Accuracy des 3 optimiseurs

## Résultats
- **Nadam** et **Adam** convergent dès l'époque 1 (100% accuracy)
- **SGD** plus lent mais converge aussi vers 100%

## Lancer le projet
```bash
py -3.11 optimizers_comparison.py
```

## Technologies
Python 3.11 | NumPy | Matplotlib | Scikit-learn
