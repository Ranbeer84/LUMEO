# Lumeo — AI-Powered Photo Memory System

> Transform your photo collection into a conversational memory you can talk to.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Node](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org/)
[![Stack](https://img.shields.io/badge/Stack-React%20%2B%20Flask-informational)](#architecture)

---

## What is Lumeo?

Lumeo is a multi-modal AI memory system that lets you search your photos using natural language — the way you actually remember them.

Instead of scrolling through folders, just ask:

```
"Show me happy moments from last summer"
"When did I meet Abhigyan at the beach?"
"Photos where I'm wearing a black t-shirt"
"My most positive memories with family"
```

Lumeo understands meaning, emotion, and context — not just keywords.

---

## Current Status

Lumeo is actively evolving from a photo organizer (v1.0) into a full conversational AI memory system.

### v1.0 — Working Now

- [x] Face detection and recognition
- [x] Automatic person clustering (DBSCAN)
- [x] Photo organization by person
- [x] React + Flask web interface
- [x] SQLite database backend

### Transformation Roadmap

| Phase | Goal                                | Status  |
| ----- | ----------------------------------- | ------- |
| 1     | SQLite → PostgreSQL + pgvector      | Planned |
| 2     | Add emotions, objects, scenes, CLIP | Planned |
| 3     | Vector + hybrid retrieval           | Planned |
| 4     | Local LLM via Ollama (Llama 3.3)    | Planned |
| 5     | Conversational chat interface       | Planned |
| 6     | Gallery → Conversational UI         | Planned |
| 7     | Insights, relationships, analytics  | Planned |
| 8     | Deployment & documentation          | Planned |

> See [VISION.md](VISION.md) for the full transformation plan.

---

## Architecture

### Current Stack (v1.0)

| Layer    | Technology                              |
| -------- | --------------------------------------- |
| Frontend | React + Vite + Lucide Icons             |
| Backend  | Flask + face_recognition + scikit-learn |
| Database | SQLite                                  |
| Storage  | Local filesystem                        |

### Target Stack

| Layer     | Technology                                  |
| --------- | ------------------------------------------- |
| Frontend  | React + Streaming Chat Interface            |
| Backend   | Flask + Multi-Modal AI Pipeline             |
| Database  | PostgreSQL + pgvector                       |
| Vision AI | face_recognition + DeepFace + YOLOv8 + CLIP |
| LLM       | Ollama — Llama 3.3 (local)                  |
| Retrieval | Hybrid RAG (semantic + keyword + filters)   |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- 8 GB+ RAM
- Git

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd lumeo
```

### 2. Start the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend runs at `http://localhost:5002`

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3002`

### 4. Use the App

1. Open `http://localhost:3002` in your browser
2. Upload photos (drag-and-drop or file picker)
3. Click **Start Face Detection**
4. Label people in each cluster
5. Click **Organize Folders** to generate organized copies

---

## Project Structure

```
lumeo/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── photo_organizer.db     # SQLite database (auto-created)
│   ├── uploads/               # Uploaded photos
│   ├── thumbnails/            # Face thumbnails
│   └── organized_photos/      # Output directory
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Root React component
│   │   ├── main.jsx           # Entry point
│   │   └── index.css          # Global styles
│   ├── package.json
│   └── vite.config.js
│
├── VISION.md                  # Full transformation roadmap
├── TRANSFORMATION_LOG.md      # Progress log
└── README.md
```

---

## Features

### v1.0 — Photo Organizer

- **Upload**: Drag-and-drop or file selection with batch support and progress tracking
- **Face Recognition**: Automatic detection using dlib's ResNet model (128-d embeddings)
- **Clustering**: DBSCAN groups faces without needing to specify the number of people
- **Organization**: Labels photos per person and copies them into named folders
- **Web UI**: Glassmorphic React interface with step-by-step workflow

### Upcoming — AI Memory System

- **Emotion Detection**: Recognize happiness, sadness, surprise, and more
- **Object Recognition**: Detect clothing, items, colors, and scenes
- **Scene Classification**: Indoor/outdoor, beach, office, events
- **Semantic Search**: CLIP embeddings for meaning-level queries
- **Natural Language Chat**: "Show me beach photos from last summer"
- **Streaming Responses**: Real-time word-by-word AI replies
- **Intelligent Insights**: Trends, relationships, and event detection

---

## Database Schema (v1.0)

```sql
photos
  photo_id    INTEGER  PRIMARY KEY
  filename    TEXT
  path        TEXT
  upload_date TEXT

clusters
  cluster_id  INTEGER  PRIMARY KEY
  name        TEXT     -- user-assigned label
  face_count  INTEGER
  thumbnail   TEXT
  created_at  TEXT

face_embeddings
  embedding_id   INTEGER  PRIMARY KEY
  photo_id       INTEGER  REFERENCES photos
  cluster_id     INTEGER  REFERENCES clusters
  embedding      BLOB     -- 128-d vector
  face_location  TEXT     -- JSON

photo_clusters                   -- many-to-many junction
  photo_id    INTEGER  REFERENCES photos
  cluster_id  INTEGER  REFERENCES clusters
```

> Note: The schema will be significantly extended during the transformation to PostgreSQL + pgvector.

---

## Safety & Rollback

A stable checkpoint exists at tag `v1.0-photo-organizer`.

```bash
# Rollback to v1.0
git checkout v1.0-photo-organizer

# Or use the rollback script
./rollback-script.sh

# Compare transformation progress
git diff v1.0-photo-organizer ai-transformation

# List all tags
git tag -l
```

---

## Technical Highlights

### ML Components

| Component        | Technology                             |
| ---------------- | -------------------------------------- |
| Face Recognition | dlib CNN encoder (128-d embeddings)    |
| Clustering       | DBSCAN — no fixed cluster count needed |
| Future Vision    | DeepFace, YOLOv8, CLIP                 |
| Future LLM       | Llama 3.3 via Ollama (fully local)     |

### Design Decisions

- **Local LLM** — Privacy-first, no API costs, runs offline
- **PostgreSQL + pgvector** — Production-ready semantic search
- **DBSCAN** — Handles unknown cluster count and noisy outliers
- **Modular services** — Each AI component is independently testable and swappable
- **RAG pattern** — Grounded responses from your actual photos, not hallucinations

---

## Contributing

Lumeo is in active development. Contributions are welcome for:

- Bug reports in v1.0
- Feature suggestions for the AI system
- Implementation help on any transformation phase

Please open an issue or submit a pull request.

---

## License

[MIT License](LICENSE) — Free to use and modify.

---

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- [Flask](https://flask.palletsprojects.com/)
- [lucide-react](https://lucide.dev/)
