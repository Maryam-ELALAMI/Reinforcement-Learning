# Reinforcement Learning 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

Ce dépôt contient les travaux pratiques de Machine Learning II sur l'apprentissage par renforcement, réalisés dans le cadre du cours à l'École Nationale de l'Intelligence Artificielle et du Digital.

## 📋 Table des matières
- [TP1: Découverte d'OpenAI Gym](#tp1-découverte-dopenai-gym)
- [TP2: Algorithmes de Base (Q-Learning/SARSA)](#tp2-algorithmes-de-base)
- [TP3: Optimisation des Feux de Circulation](#tp3-optimisation-des-feux-de-circulation)
- [TP4: PPO Avancé](#tp4-ppo-avancé)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Contribution](#-contribution)

## 🏗️ TP1: Découverte d'OpenAI Gym

### 🎯 Objectif
Prendre en main les environnements Gym et les concepts de base du Reinforcement Learning (RL).

### 📖 Définition et Rôle d'OpenAI Gym
**OpenAI Gym** est une boîte à outils standardisée pour le développement et la comparaison d'algorithmes d'apprentissage par renforcement. Son rôle principal comprend :

1. **Standardisation** :
   - Fournit une interface commune pour tous les environnements (méthodes `reset()`, `step()`)
   - Permet des comparaisons équitables entre algorithmes

2. **Bibliothèque d'environnements** :
   - Environnements classiques (CartPole, MountainCar)
   - Environnements Atari (Jeux vidéo)
   - Environnements 2D/3D de physique (MuJoCo)

3. **Outils d'évaluation** :
   - Métriques standardisées (récompense cumulée, durée des épisodes)
   - Capacité d'enregistrement des résultats

4. **Flexibilité** :
   - Prise en charge de la création d'environnements personnalisés
   - Compatibilité avec PyTorch et TensorFlow

### 🛠 Fonctionnement de Base
Le flux typique d'interaction avec Gym suit ce schéma :
