# Intégrales simples, doubles et triples : de l'aire au centre de masse

Un projet de mathématiques appliquées sous Python (NumPy, SciPy, Matplotlib) avec une vidéo animée de 16 s.

## Ce que fait chaque type d'intégrale (et ce que le projet calcule)

| Type | Sert à calculer | Cas traité | Valeur exacte |
|---|---|---|---|
| Simple ∫ | aire, longueur d'arc, travail, probabilité | ∫₀^π sin x dx par sommes de Riemann (n = 1 → 100 rectangles) | 2 |
| Double ∬ | volume sous une surface, masse d'une plaque, centre de gravité | ∬ sin x sin y dx dy sur [0,π]² par colonnes (1×1 → 20×20) | 4 |
| Triple ∭ | masse d'un solide, centre de masse, moment d'inertie | sphère de rayon 1, densité ρ = 1 + z, par Monte Carlo (N → 200 000) | M = 4π/3, z_G = 1/5, Iz = 8π/15 |

Pour la sphère : M = ∭ρ dV, z_G = (1/M)∭zρ dV, Iz = ∭(x²+y²)ρ dV.
En coordonnées sphériques, les termes impairs en z s'annulent, d'où M = 4π/3 (le volume), z_G = (4π/15)/(4π/3) = 1/5 et Iz = 8π/15.

## Méthode

- **1D** : somme de Riemann à gauche, point milieu, Simpson.
- **2D** : somme de point milieu sur une grille n × n (les colonnes de la vidéo).
- **3D** : Monte Carlo dans le cube [-1,1]³ (volume 8), estimateur = 8 × moyenne de ρ·1_{dans la sphère}. Le centre de masse est le rapport de deux estimateurs.
- **Contrôle** : toutes les valeurs exactes sont revérifiées avec `scipy.integrate` (`quad`, `dblquad`, `tplquad`).

## Résultats (`results/resultats.md`)

| Intégrale | Exact | Numérique | Erreur relative |
|---|---|---|---|
| ∫ sin (Riemann, n=100) | 2 | 1,99984 | 0,008 % |
| ∬ sin x sin y (20×20) | 4 | 4,00823 | 0,206 % |
| Masse M (Monte Carlo, N=200 000) | 4,18879 | 4,19492 | 0,146 % |
| Centre de masse z_G | 0,20000 | 0,20002 | 0,012 % |
| Moment d'inertie Iz | 1,67552 | 1,67687 | 0,081 % |

## Ordres de convergence (`results/ordres_de_convergence.png`)

Pentes mesurées sur l'erreur en échelle log-log :

| Méthode | Pente mesurée |
|---|---|
| Simpson (1D) | −4,00 |
| Point milieu, grille d=1 | −2,00 (en n) |
| Point milieu, grille d=2 | −1,00 (en N = n² évaluations) |
| Point milieu, grille d=3 | −0,67 (en N = n³ évaluations) |
| Monte Carlo d=3 | −0,50 |
<img width="1440" height="576" alt="ordres_de_convergence" src="https://github.com/user-attachments/assets/9febe42d-9db1-481b-b018-f70c90ae32f9" />

Lecture : à nombre d'évaluations N fixé, la grille perd en efficacité quand la dimension augmente (N^-2, N^-1, N^-2/3), alors que Monte Carlo reste en N^-1/2 quelle que soit la dimension. C'est ce qui le rend incontournable en grande dimension ou sur un domaine irrégulier.

Remarque : la somme de Riemann à gauche converge ici à l'ordre 2 (comme le point milieu) parce que sin s'annule aux deux bornes, ce qui la rend identique à la règle des trapèzes. Pour une fonction non nulle aux bornes, elle serait d'ordre 1.

## Limites

- Les pentes de la grille 2D/3D sont mesurées sur l'intégrande sin x sin y (sin z) sur un pavé, pas sur la sphère. La sphère, domaine irrégulier, est traitée par Monte Carlo.
- Les estimations Monte Carlo de la vidéo viennent d'une seule graine ; l'erreur varie d'un tirage à l'autre (en 1/√N).
- La comparaison à Monte Carlo n'utilise ni réduction de variance ni quasi-Monte Carlo (pistes possibles).

## Lancer

    pip install numpy scipy matplotlib      # + ffmpeg installé
    python integrales_multiples.py

Produit dans `results/` : `integrales_multiples.mp4`
