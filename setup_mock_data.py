"""
Setup script to create mock data for demonstration and testing.

This script creates sample video transcripts and PDF files in the data/ directory.
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_mock_video_transcripts():
    """Create mock video transcript JSON files."""
    
    videos_dir = Path("data/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Video 1: Kubernetes Setup Tutorial
    video1 = {
        "video_id": "VIDEO_KUBERNETES_SETUP",
        "title": "How to Setup Kubernetes Cluster",
        "pdf_reference": "kubernetes_guide.pdf",
        "duration_seconds": 300,
        "video_transcripts": [
            {"id": i, "timestamp": i * 0.5, "word": word}
            for i, word in enumerate([
                "Hello", "everyone", "today", "I", "will", "teach", "you",
                "how", "to", "setup", "Kubernetes", "cluster", "First",
                "you", "need", "to", "install", "Docker", "on", "your",
                "server", "Then", "enable", "kubeadm", "kubectl", "and",
                "kubelet", "Make", "sure", "your", "server", "has",
                "at", "least", "two", "CPUs", "and", "two", "gigabytes",
                "of", "RAM", "Next", "initialize", "the", "control",
                "plane", "with", "kubeadm", "init", "This", "will",
                "generate", "a", "join", "token", "for", "worker",
                "nodes", "Copy", "this", "token", "to", "your", "clipboard"
            ])
        ]
    }
    
    with open(videos_dir / "kubernetes_setup.json", "w") as f:
        json.dump(video1, f, indent=2)
    
    # Video 2: Docker Basics
    video2 = {
        "video_id": "VIDEO_DOCKER_BASICS",
        "title": "Docker Fundamentals",
        "pdf_reference": "docker_guide.pdf",
        "duration_seconds": 240,
        "video_transcripts": [
            {"id": i, "timestamp": i * 0.5, "word": word}
            for i, word in enumerate([
                "Docker", "is", "a", "containerization", "platform",
                "that", "simplifies", "application", "deployment",
                "Containers", "are", "lightweight", "and", "portable",
                "You", "can", "run", "the", "same", "container",
                "on", "any", "system", "A", "Docker", "image", "is",
                "a", "blueprint", "for", "creating", "containers",
                "Think", "of", "it", "like", "a", "class", "in",
                "object", "oriented", "programming", "The", "Dockerfile",
                "is", "where", "you", "define", "your", "image",
                "It", "contains", "instructions", "for", "building",
                "the", "image", "Common", "commands", "include", "FROM",
                "RUN", "COPY", "and", "CMD"
            ])
        ]
    }
    
    with open(videos_dir / "docker_basics.json", "w") as f:
        json.dump(video2, f, indent=2)
    
    print(f"Created {len([video1, video2])} mock video transcripts")


def create_mock_pdfs():
    """Create mock PDF files with sample content."""
    
    pdfs_dir = Path("data/pdfs")
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Kubernetes Guide PDF
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        pdf_path = pdfs_dir / "kubernetes_guide.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        
        y = 750
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Kubernetes Setup Guide")
        
        y -= 30
        c.setFont("Helvetica", 11)
        
        content = [
            "What is Kubernetes?\n",
            "Kubernetes is an open-source container orchestration platform that automates many of the manual processes involved in deploying, managing, and scaling containerized applications. It groups containers that make up an application into logical units for easy management and discovery.\n",
            "\nInstallation Requirements\n",
            "Before installing Kubernetes, ensure your system meets these requirements: Linux operating system (preferably Ubuntu 18.04 or later), at least 2 CPU cores, at least 2GB of RAM, and unique hostname for each node in the cluster.\n",
            "\nStep 1: Install Docker\n",
            "First, install Docker as Kubernetes typically runs containers using Docker. Update your package manager, then install Docker from the official repository. Make sure to add your user to the docker group to avoid using sudo.\n",
            "\nStep 2: Install kubeadm, kubelet, and kubectl\n",
            "These are essential components. kubeadm is used to bootstrap the cluster, kubelet is the agent that ensures containers are running in pods, and kubectl is the command-line tool for managing the cluster.\n",
            "\nStep 3: Initialize the Control Plane\n",
            "Use kubeadm init to set up the control plane node. This will download container images and configure the cluster. The output will include a kubeadm join command that you use to add worker nodes to the cluster.\n",
        ]
        
        for line in content:
            if line.strip():
                lines = [line[i:i+80] for i in range(0, len(line), 80)]
                for sub_line in lines:
                    c.drawString(50, y, sub_line)
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = 750
        
        c.save()
        print(f"Created PDF: {pdf_path}")
    except ImportError:
        print("reportlab not installed, skipping PDF creation")


def create_mock_data():
    """Create all mock data."""
    print("Creating mock data for RAG chatbot...")
    create_mock_video_transcripts()
    try:
        create_mock_pdfs()
    except Exception as e:
        print(f"Could not create PDFs: {e}")
    
    print("Mock data creation complete!")


if __name__ == "__main__":
    create_mock_data()
