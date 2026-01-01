⸻

Lumeo — AI-Powered Photo Memory System

Transforming personal photos into conversational memories using multi-modal AI

⸻

Vision

Lumeo is an AI-powered conversational photo memory system that understands photos the way humans remember them — through language, emotions, people, and context, not folders.

Instead of browsing directories, users talk to their photo collection.

Ask naturally:
	•	“Show me happy moments from last summer”
	•	“When did I meet Abhigyan at the beach?”
	•	“Photos where I’m wearing a black t-shirt”

Lumeo responds with:
	•	Semantic understanding beyond keywords
	•	Emotion and mood awareness
	•	Relationship and timeline-based insights

⸻

Project Status

Current Phase: Phase 0 → Multi-Modal RAG Transformation
Stable Baseline: v1.0-photo-organizer

What’s Working Now (v1.0)
	•	Face detection and recognition
	•	Automatic person clustering (DBSCAN)
	•	Photo organization by person
	•	React + Flask web application
	•	SQLite backend

⸻

Transformation Roadmap

Lumeo is evolving from a photo organizer into a conversational AI memory system:
	1.	Database Evolution — PostgreSQL + pgvector
	2.	Vision Intelligence — emotions, objects, scenes, CLIP
	3.	Retrieval System — vector + hybrid search
	4.	Generation Layer — local LLM (Ollama, Llama 3.3)
	5.	Conversational Memory — context-aware chat
	6.	Frontend Transformation — gallery → conversational UI
	7.	Insights & Analytics
	8.	Deployment & Documentation

See VISION.md for the detailed plan.

⸻

Architecture

Current Stack (v1.0)

Frontend:  React (Vite-based)
Backend:   Flask + face_recognition + scikit-learn
Database:  SQLite
Storage:   Local filesystem

Target Stack

Frontend:  React + Streaming Chat Interface
Backend:   Flask + Multi-Modal AI Pipeline
Database:  PostgreSQL + pgvector for semantic search
Vision AI: face_recognition + DeepFace + YOLOv8 + CLIP
LLM:       Ollama (Llama 3.3) - Local LLM
RAG:       Hybrid retrieval (semantic + filters)


⸻

Setup (v1.0)

Prerequisites
	•	Python 3.11+
	•	Node.js 18+
	•	Git

Run Locally

Backend

cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Backend → http://localhost:5002

Frontend

cd frontend
npm install
npm run dev

Frontend → http://localhost:3002

⸻

Planned Capabilities
	•	Emotion, object, and scene detection
	•	Semantic image understanding with CLIP
	•	Natural language photo queries
	•	Context-aware conversational memory
	•	Intelligent insights (relationships, trends, events)

⸻

Safety & Rollback

A stable checkpoint exists at:

v1.0-photo-organizer

Rollback instantly

./scripts/rollback.sh

Compare changes

git diff v1.0-photo-organizer ai-transformation


⸻

Technical Highlights
	•	Face Recognition: dlib ResNet (128-D embeddings)
	•	Clustering: DBSCAN (no fixed cluster count)
	•	Architecture: Modular, pipeline-based AI system
	•	RAG Pattern: Retrieval-augmented, grounded responses
	•	Local LLM: Privacy-first, cost-efficient design

⸻

📄 License

MIT License

⸻

From simple photo organization to conversational AI memory — this is Lumeo. 🚀

⸻