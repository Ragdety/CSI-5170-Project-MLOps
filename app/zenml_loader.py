
from zenml.client import Client

# Get the active stack
client = Client()

def load_artifact_by_version(artifact_id):
  artifact = client.get_artifact_version(artifact_id)
  return artifact.load()

