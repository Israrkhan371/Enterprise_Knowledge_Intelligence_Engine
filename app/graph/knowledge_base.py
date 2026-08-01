"""
Curated domain knowledge used to bootstrap technology-ecosystem grouping,
skill-prerequisite ordering, and relationship-type inference in
app/graph/relationships.py.

This is a static, hand-maintained knowledge base rather than an LLM call.
Relationship inference has to run on every document ingested (including
bulk GitHub repo ingests - dozens of files per call, see
ingest_github_repo() in app/ingestion/pipeline.py), so it needs to be
fast, deterministic and free. A curated baseline covers the common
enterprise stack well; anything not covered here falls back to pure
evidence-based inference (see relationships.infer_relationship).
"""

# Groups technologies into ecosystems (case study Step 5). A technology
# appears in exactly one ecosystem unless there's a real justification to
# split it, per the instructions ("Do not place technologies in multiple
# ecosystems unless justified").
TECHNOLOGY_ECOSYSTEMS: dict[str, list[str]] = {
    "Python": ["FastAPI", "Django", "Flask", "SQLAlchemy", "Pandas", "NumPy",
               "LangChain", "LlamaIndex", "PyTest", "PyTorch", "TensorFlow", "spaCy"],
    "JavaScript": ["TypeScript", "Node.js", "React", "Next.js", "Redux", "TailwindCSS", "Express"],
    "Containers & Orchestration": ["Docker", "Docker Compose", "Kubernetes", "Nginx"],
    "Databases": ["PostgreSQL", "MongoDB", "Redis", "Neo4j", "Elasticsearch", "ChromaDB"],
    "Cloud Platforms": ["AWS", "Azure", "GCP"],
    "Infrastructure as Code": ["Terraform"],
    "Observability": ["Prometheus", "Grafana", "MLflow"],
    "Systems Languages": ["Rust", "Go", "Java"],
}

# Baseline known relationships: (source, target) -> (relation_type,
# base_confidence 0-100, reason). These are established facts about the
# technologies themselves, independent of how often any given document
# set happens to mention them together. Actual observed co-occurrence
# (relationships.py) adjusts confidence up/down from this baseline rather
# than replacing it - see infer_relationship().
KNOWN_RELATIONS: dict[tuple[str, str], tuple[str, int, str]] = {
    ("Python", "FastAPI"): ("PREREQUISITE_OF", 95, "FastAPI is a Python web framework and cannot be used without Python knowledge."),
    ("Python", "Django"): ("PREREQUISITE_OF", 95, "Django is a Python web framework."),
    ("Python", "Flask"): ("PREREQUISITE_OF", 93, "Flask is a Python micro web framework."),
    ("Python", "SQLAlchemy"): ("PREREQUISITE_OF", 90, "SQLAlchemy is a Python ORM/toolkit."),
    ("Python", "Pandas"): ("PREREQUISITE_OF", 90, "Pandas is a Python data-analysis library."),
    ("Python", "LangChain"): ("PREREQUISITE_OF", 90, "LangChain is most commonly used from its Python API."),
    ("Python", "PyTorch"): ("PREREQUISITE_OF", 92, "PyTorch is a Python-first deep-learning framework."),
    ("Python", "TensorFlow"): ("PREREQUISITE_OF", 92, "TensorFlow's primary API surface is Python."),
    ("Python", "spaCy"): ("PREREQUISITE_OF", 90, "spaCy is a Python NLP library."),
    ("Python", "PyTest"): ("PREREQUISITE_OF", 88, "PyTest is a Python testing framework."),
    ("FastAPI", "Docker"): ("DEPLOYS_TO", 80, "FastAPI services are commonly containerized with Docker for deployment."),
    ("JavaScript", "React"): ("PREREQUISITE_OF", 93, "React is a JavaScript UI library."),
    ("JavaScript", "TypeScript"): ("EXTENDS", 85, "TypeScript is a typed superset of JavaScript."),
    ("React", "Next.js"): ("EXTENDS", 88, "Next.js is a React framework."),
    ("React", "Redux"): ("USES", 75, "Redux is a state-management library commonly paired with React."),
    ("Docker", "Kubernetes"): ("PREREQUISITE_OF", 88, "Kubernetes orchestrates containers; container fundamentals precede orchestration."),
    ("Docker", "Docker Compose"): ("EXTENDS", 85, "Docker Compose is a multi-container orchestration layer on top of Docker."),
    ("Docker Compose", "Kubernetes"): ("ALTERNATIVE_TO", 70, "Both orchestrate multi-container deployments; Kubernetes is the production-scale alternative."),
    ("Kubernetes", "Nginx"): ("USES", 60, "Nginx is commonly deployed as an ingress controller in Kubernetes clusters."),
    ("FastAPI", "PostgreSQL"): ("CONNECTS_TO", 75, "FastAPI services commonly connect to PostgreSQL via an ORM."),
    ("SQLAlchemy", "PostgreSQL"): ("CONNECTS_TO", 85, "SQLAlchemy is a common ORM layer for PostgreSQL."),
    ("Python", "Neo4j"): ("CONNECTS_TO", 70, "The official neo4j Python driver connects application code to a Neo4j database."),
    ("LangChain", "ChromaDB"): ("CONNECTS_TO", 78, "LangChain commonly integrates with ChromaDB as a vector store."),
    ("Python", "ChromaDB"): ("CONNECTS_TO", 70, "ChromaDB is accessed via its Python client library."),
    ("PyTorch", "TensorFlow"): ("ALTERNATIVE_TO", 80, "Both are deep-learning frameworks solving the same problem space."),
    ("Django", "Flask"): ("ALTERNATIVE_TO", 75, "Both are Python web frameworks solving the same problem space."),
    ("AWS", "Azure"): ("ALTERNATIVE_TO", 80, "Both are competing cloud platforms."),
    ("AWS", "GCP"): ("ALTERNATIVE_TO", 80, "Both are competing cloud platforms."),
    ("Azure", "GCP"): ("ALTERNATIVE_TO", 78, "Both are competing cloud platforms."),
    ("Terraform", "AWS"): ("DEPLOYS_TO", 78, "Terraform is commonly used to provision AWS infrastructure."),
    ("Terraform", "Azure"): ("DEPLOYS_TO", 75, "Terraform is commonly used to provision Azure infrastructure."),
    ("Terraform", "GCP"): ("DEPLOYS_TO", 75, "Terraform is commonly used to provision GCP infrastructure."),
    ("Prometheus", "Grafana"): ("USES", 82, "Grafana commonly visualizes metrics collected by Prometheus."),
    ("Docker", "Prometheus"): ("MONITORS", 55, "Prometheus is commonly deployed to monitor containerized services."),
    ("MLflow", "Python"): ("RELATED_TO", 60, "MLflow's primary client library is Python."),
}

# Skill prerequisite chain (linear MVP ordering, case study Step 6). Used
# as the backbone for skill-dependency inference, and to sanity-check that
# an inferred PREREQUISITE_OF edge doesn't contradict a well-established
# learning order ("avoid creating circular dependencies").
SKILL_PREREQUISITE_CHAIN: list[str] = [
    "Programming Fundamentals",
    "Python",
    "Object-Oriented Programming",
    "REST APIs",
    "FastAPI",
    "Docker",
    "Kubernetes",
    "Microservices",
]
