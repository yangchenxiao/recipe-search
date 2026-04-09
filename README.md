# 🍳 Recipe Search System

This project is a recipe search system that compares **traditional keyword-based retrieval (Classic IR)** with LLM-based intelligent search.

The system allows users to input natural language queries (e.g., *"quick healthy chicken dinner"*) and retrieves relevant recipes using two different approaches.

---

## 🚀 Features

### 1. Classic IR (Keyword Search)
- Uses keyword matching
- Fast and deterministic
- Limited understanding of user intent

### 2. Intelligent Search (LLM-based)
- Uses an LLM to parse user queries
- Extracts structured constraints:
  - ingredients
  - cooking time
  - course (e.g., dinner)
  - health goals
- Applies:
  - soft filtering
  - semantic matching
  - ranking optimization
- Provides **AI-generated explanations**

### 3. Compare Mode
- Displays results from both methods side-by-side
- Helps evaluate retrieval quality

---
