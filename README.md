# HAX907X

Ce dépôt contient le travail pratique consacré à l'exploration des Support Vector Machines (SVM).
Le Code est disponible dans \Code , et le projet quarto avec fichier et image qui ont généré le rapport sont disponible dans \Rendu.

Dans ce TP, on a suivi les scripts de référence pour explorer l'impact de la régularisation, des variables de nuisance et de la réduction de dimension via l'Analyse en Composantes Principales (PCA) sur les performances d'un modèle SVM.

Des modifications ont été apportées aux scripts originaux, et les justifications de ces ajustements sont discutées en fin du rapport, et en en commentaire du code lui même.

## Les principaux objectifs de ce travail pratique sont :

  * Analyser l'influence du paramètre de régularisation C sur les performances des SVM.
  
  * Évaluer l'impact des variables de nuisance sur le modèle, notamment en ajoutant du bruit aux données.
  
  * Utiliser la PCA pour réduire la dimensionnalité des données et observer son effet sur les performances du modèle.

## Méthodologie :

  * Analyse de l'influence du paramètre C : On a effectué une exploration de C sur une échelle logarithmique et observé son impact sur le score de prédiction.

  * Ajout de variables de nuisance : On a entraîné le modèle sur des données bruitées pour évaluer comment l'augmentation du bruit affecte les performances.

  * Application de la PCA : On a réduit la dimensionnalité des données bruitées afin d'analyser l'amélioration potentielle des performances du modèle.

## Résultats :

Les résultats montrent que :

  * Le choix du paramètre C a un impact significatif sur la performance du SVM.
  
  * Le nombre de variables de nuisance a moins d'impact sur les scores de prédiction que leur variance.
  
  * L'utilisation de la PCA sur les données bruitées permet d'améliorer significativement les performances du modèle.
