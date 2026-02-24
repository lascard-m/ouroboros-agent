import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le répertoire courant au path pour importer ouroboros
sys.path.insert(0, os.getcwd())

from ouroboros.llm import LLMClient

def test_provider_configuration():
    print("=== Test de configuration Multi-Provider LLM ===")
    
    # Test 1: Configuration par défaut (ou celle du .env)
    client = LLMClient()
    print(f"Provider actuel: {client.provider}")
    print(f"Base URL: {client._base_url}")
    
    # Test 2: Simulation Ollama
    print("\n--- Simulation Ollama ---")
    os.environ["LLM_PROVIDER"] = "ollama"
    client_ollama = LLMClient()
    print(f"Provider: {client_ollama.provider}")
    print(f"Base URL: {client_ollama._base_url}")
    assert client_ollama.provider == "ollama"
    
    # Test 3: Simulation OpenAI
    print("\n--- Simulation OpenAI ---")
    os.environ["LLM_PROVIDER"] = "openai"
    client_openai = LLMClient()
    print(f"Provider: {client_openai.provider}")
    print(f"Base URL: {client_openai._base_url}")
    assert client_openai.provider == "openai"
    
    # Test 4: Simulation Mistral
    print("\n--- Simulation Mistral ---")
    os.environ["LLM_PROVIDER"] = "mistral"
    client_mistral = LLMClient()
    print(f"Provider: {client_mistral.provider}")
    print(f"Base URL: {client_mistral._base_url}")
    assert client_mistral.provider == "mistral"

    print("\n✅ Tous les tests de configuration sont passés avec succès !")

if __name__ == "__main__":
    test_provider_configuration()
