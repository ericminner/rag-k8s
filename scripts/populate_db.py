import requests
import json
import os
from typing import List, Dict

# Sample Kubernetes concepts for our RAG system
KUBERNETES_DOCS = [
    {
        "title": "Pods",
        "content": """A Pod is the smallest deployable unit in Kubernetes. A Pod represents a single instance of a running process in your cluster. 
        Pods contain one or more containers, such as Docker containers. When a Pod runs multiple containers, the containers are managed as a single entity and share the Pod's resources.
        Key features include shared storage and network resources, and a specification for how to run the containers."""
    },
    {
        "title": "Deployments",
        "content": """A Deployment provides declarative updates for Pods and ReplicaSets. You describe a desired state in a Deployment, 
        and the Deployment Controller changes the actual state to the desired state at a controlled rate. You can define Deployments to 
        create new ReplicaSets, or to remove existing Deployments and adopt all their resources with new Deployments."""
    },
    {
        "title": "Services",
        "content": """A Kubernetes Service is an abstraction which defines a logical set of Pods and a policy by which to access them. 
        Services enable a loose coupling between dependent Pods. A Service routes traffic across a set of Pods. Services are the 
        abstraction that allows pods to die and replicate in Kubernetes without impacting your application."""
    },
    {
        "title": "ConfigMaps",
        "content": """ConfigMaps allow you to decouple configuration artifacts from image content to keep containerized applications portable. 
        The ConfigMap API resource stores configuration data as key-value pairs. The data can be consumed in pods or used to store 
        configuration data for system components such as controllers."""
    },
    {
        "title": "Secrets",
        "content": """Secrets let you store and manage sensitive information, such as passwords, OAuth tokens, and ssh keys. 
        Storing confidential information in a Secret is safer and more flexible than putting it verbatim in a Pod definition 
        or in a container image."""
    },
    {
        "title": "Volumes",
        "content": """On-disk files in a container are ephemeral, which presents problems for non-trivial applications when running in containers. 
        A Kubernetes volume has an explicit lifetime - the same as the Pod that encloses it. A volume outlives any containers that run 
        within the Pod, and data is preserved across container restarts."""
    },
    {
        "title": "Ingress",
        "content": """Ingress exposes HTTP and HTTPS routes from outside the cluster to services within the cluster. Traffic routing is 
        controlled by rules defined on the Ingress resource. An Ingress may be configured to give Services externally-reachable URLs, 
        load balance traffic, terminate SSL/TLS, and offer name-based virtual hosting."""
    }
]


def populate_database():
    # Using the embedding service we already created
    embedding_service_url = "http://localhost:8000/embed"

    for doc in KUBERNETES_DOCS:
        payload = {
            "title": doc["title"],
            "content": doc["content"],
            "metadata": {
                "source": "kubernetes_docs",
                "category": "technical"
            }
        }

        try:
            response = requests.post(
                embedding_service_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print(f"Successfully added document: {doc['title']}")
            else:
                print(f"Failed to add document: {doc['title']}")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error adding document {doc['title']}: {str(e)}")


if __name__ == "__main__":
    populate_database()