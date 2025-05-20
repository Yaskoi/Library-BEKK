# BEKK-GARCH Python Library

Une implémentation en Python du modèle **GARCH multivarié BEKK (p,q)**, conçu pour modéliser les covariances conditionnelles entre plusieurs séries temporelles financières.

## 📈 Objectif

Le modèle BEKK permet :
- De capturer la **dynamique de la volatilité croisée** entre actifs,
- D'assurer la **positivité définie** de la matrice de covariance conditionnelle,
- D'être utilisé en **gestion de portefeuille**, **modélisation du risque** ou **arbitrage statistique**.

---

## ⚙️ Fonctionnalités

- Estimation de BEKK(p,q) pleine pour 2 actifs
- Optimisation par maximum de vraisemblance (`scipy.optimize`)
- Génération des **covariances conditionnelles**
- Visualisation automatique des **variances et covariances conditionnelles**

---

## 🧪 Installation

```bash
pip install git+https://github.com/Yaskoi/ysk_garch.git

