import os
from google import genai

api_key = 'AIzaSyAiO_orZ5ZdCuFlpwYeM_QyrHBsf8_uvqQ'

client = genai.Client(api_key=api_key)

print("Available models that support generateContent:\n")
for model in client.models.list():
    # Only show models that support content generation
    if hasattr(model, 'supported_actions') and 'generateContent' in (model.supported_actions or []):
        print(f"  {model.name}")
    else:
        print(f"  {model.name}  (actions: {getattr(model, 'supported_actions', 'unknown')})")