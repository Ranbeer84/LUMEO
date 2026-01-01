# Lumeo - AI-Powered Photo Memory System

> Transforming photos into conversational memories through multi-modal AI
> 

## 🎯 Vision

Lumeo is an **AI-powered conversational photo memory system** that understands your photos the way you remember them - through natural language, emotions, and context. Instead of organizing photos into folders, Lumeo lets you *talk* to your photo collection.

**Ask naturally:**

- "Show me happy moments from last summer"
- "When did I meet Abhigyan at the beach?"
- "Photos where I'm wearing a black t-shirt"
- "My most positive memories with family"

**Get intelligent responses:**

- Semantic understanding beyond keywords
- Emotion and mood analysis
- Relationship detection and insights
- Timeline of memories with context

---

## 🚀 Current Status: Phase 0 → Multi-Modal RAG Transformation

### ✅ What's Working Now (v1.0 - Photo Organizer)

- Face detection and recognition
- Automatic person clustering (DBSCAN)
- Photo organization by person
- Web interface (React + Flask)
- SQLite database backend

### 🎯 Transformation Roadmap

This project is evolving from a simple photo organizer into a sophisticated AI memory system:

**Phase 1:** Database Evolution (SQLite → PostgreSQL + pgvector)

**Phase 2:** Vision Intelligence (Faces → Emotions + Objects + Scenes + CLIP)

**Phase 3:** Retrieval System (Vector search + Hybrid queries)

**Phase 4:** Generation Layer (Local LLM with Ollama + Llama 3.3)

**Phase 5:** Conversational Memory (Chat interface with context)

**Phase 6:** Frontend Transformation (Gallery → Conversational UI)

**Phase 7:** Intelligence & Polish (Insights, relationships, analytics)

**Phase 8:** Deployment & Documentation

> See VISION.md for the complete transformation plan
> 

---

## 🏗️ Architecture

### Current Stack (v1.0)

```
Frontend:  React + Vite + Lucide Icons
Backend:   Flask + face_recognition + scikit-learn
Database:  SQLite
Storage:   Local filesystem

```

### Target Stack (After Transformation)

```
Frontend:  React + Streaming Chat Interface
Backend:   Flask + Multi-Modal AI Pipeline
Database:  PostgreSQL + pgvector for semantic search
Vision AI: face_recognition + DeepFace + YOLOv8 + CLIP
LLM:       Ollama (Llama 3.3) - Local LLM
RAG:       Hybrid retrieval (semantic + filters)

```

---

## 🛠️ Current Setup (v1.0)

### Prerequisites

- Python 3.11+
- Node.js 18+
- 8GB+ RAM
- Git

### Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd lumeo

```

**2. Backend Setup**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py

```

Backend runs on `http://localhost:5002`

**3. Frontend Setup**

```bash
cd frontend
npm install
npm run dev

```

Frontend runs on `http://localhost:3002`

**4. Upload & Organize Photos**

- Open browser to http://localhost:3002
- Upload multiple photos
- Click "Start Face Detection"
- Label people in clusters
- Click "Organize Folders" to create organized copies


## 🧪 Current Features (v1.0)

### Photo Upload & Processing

- Drag-and-drop or file selection
- Batch upload support
- Progress tracking

### Face Recognition

- Automatic face detection in photos
- Face encoding using dlib's ResNet model
- DBSCAN clustering to group same people
- Quality-based thumbnail selection

### Organization

- Label people in clusters
- Automatic folder creation per person
- Photos copied to respective person folders
- Preserves original files

### Web Interface

- Modern glassmorphic design
- Responsive layout (mobile-friendly)
- Step-by-step workflow
- Cluster photo viewer with navigation

---

## 🔮 Upcoming Features (Transformation)

### Multi-Modal Understanding

- **Emotion Detection**: Recognize happiness, sadness, surprise, etc.
- **Object Recognition**: Detect clothing, items, colors
- **Scene Classification**: Indoor/outdoor, beach/office, etc.
- **CLIP Embeddings**: Semantic understanding of image content

### Conversational Interface

- **Natural Language Queries**: "Show me beach photos from last summer"
- **Context-Aware Chat**: Follow-up questions work naturally
- **Streaming Responses**: Real-time AI responses word-by-word
- **Photo Inline Display**: Photos appear in conversation flow

### Intelligent Insights

- "You appear happiest at the beach"
- "Your most frequent companion is [person]"
- Emotional trends over time
- Event detection (birthdays, vacations)

### Advanced Retrieval

- Semantic search (meaning, not just keywords)
- Hybrid queries (concepts + filters + people)
- Similarity search ("photos like this one")
- Multi-criteria ranking

---

## 🎓 Technical Highlights (For Thesis/Interviews)

### Machine Learning Components

1. **Face Recognition**: dlib CNN face encoder (128-d embeddings)
2. **Clustering**: DBSCAN with adaptive parameters
3. **Future**: DeepFace, YOLOv8, CLIP, Llama 3.3

### System Design

- **Modular Architecture**: Separate services for each AI component
- **Pipeline Orchestration**: Coordinated multi-model processing
- **RAG Pattern**: Retrieval-Augmented Generation for grounded responses
- **Vector Search**: pgvector for semantic similarity

### Key Decisions

- **Local LLM**: Privacy + Cost + Learning (vs OpenAI API)
- **PostgreSQL**: Production-ready + pgvector support
- **DBSCAN**: Handles unknown cluster count + outliers
- **Modular Services**: Easy to test, swap, and scale

---

## 🔒 Safety & Rollback (v1.0 Checkpoint)

A safety checkpoint has been created at tag `v1.0-photo-organizer`.

**To rollback to v1.0:**

```bash
./rollback-script.sh
# or manually:
git checkout v1.0-photo-organizer

```

**To see transformation progress:**

```bash
git diff v1.0-photo-organizer ai-transformation

```

**To list all tags:**

```bash
git tag -l

```

---

## 📊 Current Database Schema (v1.0)

```sql
photos
  - photo_id (PK)
  - filename
  - path
  - upload_date

clusters (people)
  - cluster_id (PK)
  - name (user-assigned label)
  - face_count
  - thumbnail
  - created_at

face_embeddings
  - embedding_id (PK)
  - photo_id (FK)
  - cluster_id (FK)
  - embedding (blob: 128-d vector)
  - face_location (JSON)

photo_clusters (junction table)
  - photo_id (FK)
  - cluster_id (FK)
  - (handles many-to-many: one photo can have multiple people)

```

> Note: Schema will be dramatically enhanced during transformation
> 

---

## 🤝 Contributing

This project is currently in active transformation. If you'd like to:

- Report bugs in v1.0
- Suggest features for the AI system
- Contribute to the transformation

Please open an issue or PR!

---

## 📄 License

MIT License - Feel free to use and modify

---

## 🙏 Acknowledgments

Built with:

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- [Flask](https://flask.palletsprojects.com/)
- [lucide-react](https://lucide.dev/) for beautiful icons

Transformation inspired by modern RAG systems and conversational AI.

---

## 📞 Contact

Questions about the transformation or technical approach? Open an issue!

---

**From simple photo organizer to conversational AI memory system - Join the journey! 🚀**
