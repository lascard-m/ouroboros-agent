# Dockerfile pour Ouroboros en mode bac à sable
FROM python:3.10-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash ouroboros
USER ouroboros

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY --chown=ouroboros:ouroboros . .

# Installer Python dependencies
RUN pip install --no-cache-dir -e .

# Variables d'environnement pour le bac à sable
ENV SANDBOX_MODE=true
ENV TOTAL_BUDGET=10
ENV OUROBOROS_MAX_WORKERS=2
ENV OUROBOROS_MAX_ROUNDS=50

# Point d'entrée
CMD ["python", "local_launcher.py"]
