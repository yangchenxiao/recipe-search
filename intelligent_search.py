from dotenv import load_dotenv
load_dotenv()
import os
import json
import ast
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/recipes_project_subset_4000.csv"


# =========================
# Basic Utilities
# =========================
def parse_list_field(x):
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []

        try:
            value = json.loads(x)
            if isinstance(value, list):
                return value
        except Exception:
            pass

        try:
            value = ast.literal_eval(x)
            if isinstance(value, list):
                return value
        except Exception:
            pass

        return [x]

    return []


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def course_aliases(course_norm):
    """
    Map user-friendly meal words to dataset-style course labels.
    """
    mapping = {
        "dinner": ["main"],
        "lunch": ["main", "salad", "soup"],
        "breakfast": ["breakfast", "brunch"],
        "dessert": ["dessert"],
        "snack": ["appetizer", "snack"],
        "starter": ["appetizer", "side"],
        "appetizer": ["appetizer", "side"],
        "main": ["main"],
        "soup": ["soup"],
        "drink": ["drink", "beverage"],
        "beverage": ["drink", "beverage"],
        "brunch": ["brunch", "breakfast"],
    }
    return mapping.get(course_norm, [course_norm])


def keyword_overlap_score(text, terms):
    if not terms:
        return 0.0
    text = normalize_text(text)
    hits = sum(1 for t in terms if t in text)
    return hits / max(len(terms), 1)


# =========================
# Data Loading
# =========================
def build_recipe_context(row):
    ingredients_text = ", ".join(row["ingredients"]) if isinstance(row["ingredients"], list) else ""
    directions_text = " ".join(row["directions"][:3]) if isinstance(row["directions"], list) else ""

    return f"""
Title: {row.get('recipe_title', '')}
Description: {row.get('description', '')}
Cuisine: {row.get('cuisine_primary', '')}
Course: {row.get('course_primary', '')}
Main Ingredient: {row.get('main_ingredient', '')}
Difficulty: {row.get('difficulty', '')}
Health Level: {row.get('health_level', '')}
Healthiness Score: {row.get('healthiness_score', '')}
Total Time: {row.get('total_time_min', '')} min
Ingredients: {ingredients_text}
Directions: {directions_text}
""".strip()


@lru_cache(maxsize=1)
def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    list_columns = [
        "ingredients",
        "directions",
        "ingredients_canonical",
        "cuisine_list",
        "course_list",
        "tastes",
        "dietary_profile",
        "health_flags"
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_field)

    # Normalized text fields
    df["recipe_title_norm"] = df["recipe_title"].fillna("").apply(normalize_text)
    df["description_norm"] = df["description"].fillna("").apply(normalize_text)
    df["course_primary_norm"] = df["course_primary"].fillna("").apply(normalize_text)
    df["cuisine_primary_norm"] = df["cuisine_primary"].fillna("").apply(normalize_text)
    df["main_ingredient_norm"] = df["main_ingredient"].fillna("").apply(normalize_text)
    df["difficulty_norm"] = df["difficulty"].fillna("").apply(normalize_text)
    df["health_level_norm"] = df["health_level"].fillna("").apply(normalize_text)

    df["ingredients_norm"] = df["ingredients"].apply(
        lambda vals: [normalize_text(v) for v in vals if normalize_text(v) != ""]
    )
    df["ingredients_text_norm"] = df["ingredients_norm"].apply(lambda vals: " | ".join(vals))

    # Safe numeric
    if "total_time_min" in df.columns:
        df["total_time_min"] = pd.to_numeric(df["total_time_min"], errors="coerce")
    else:
        df["total_time_min"] = np.nan

    if "healthiness_score" in df.columns:
        df["healthiness_score"] = pd.to_numeric(df["healthiness_score"], errors="coerce")
    else:
        df["healthiness_score"] = np.nan

    # Combined text for semantic context
    df["semantic_text"] = df.apply(build_recipe_context, axis=1)

    return df


# =========================
# Embedding Model + Matrix
# =========================
@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_recipe_embeddings():
    df = load_data(DATA_PATH)
    model = get_embedding_model()
    texts = df["semantic_text"].tolist()

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return embeddings


# =========================
# OpenAI Client
# =========================
@lru_cache(maxsize=1)
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def call_llm(messages, model="gpt-4o-mini", temperature=0):
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content


# =========================
# Query Parsing
# =========================
QUERY_PARSER_SYSTEM_PROMPT = """
You are a recipe search assistant.

Convert the user's natural-language recipe request into a JSON object.
Return ONLY valid JSON.

Required JSON keys:
- include_ingredients: list of strings
- exclude_ingredients: list of strings
- max_time: integer or null
- course: string or null
- cuisine: string or null
- dietary_constraints: list of strings
- health_goal: string or null
- style_keywords: list of strings
- semantic_query: string
- keyword_query: string

Rules:
1. Use null if information is not clearly specified.
2. Keep ingredient names short and normalized.
3. dietary_constraints may include:
   vegan, vegetarian, gluten_free, dairy_free, nut_free, halal, kosher
4. semantic_query should be a concise natural-language retrieval query.
5. keyword_query should be a short keyword-style query.
6. Use course only if clearly implied, e.g. breakfast, dinner, dessert.
7. health_goal may be values like healthy, low_calorie, high_protein if clearly implied.
8. Return JSON only.
""".strip()


def parse_nl_query(user_query, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": QUERY_PARSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    raw_output = call_llm(messages, model=model, temperature=0)

    try:
        parsed = json.loads(raw_output)
    except Exception:
        # Safe fallback
        parsed = {
            "include_ingredients": [],
            "exclude_ingredients": [],
            "max_time": None,
            "course": None,
            "cuisine": None,
            "dietary_constraints": [],
            "health_goal": None,
            "style_keywords": [],
            "semantic_query": user_query,
            "keyword_query": user_query
        }

    # Final cleanup / normalization
    parsed["include_ingredients"] = [
        normalize_text(x) for x in parsed.get("include_ingredients", []) if normalize_text(x)
    ]
    parsed["exclude_ingredients"] = [
        normalize_text(x) for x in parsed.get("exclude_ingredients", []) if normalize_text(x)
    ]
    parsed["course"] = normalize_text(parsed["course"]) if parsed.get("course") else None
    parsed["cuisine"] = normalize_text(parsed["cuisine"]) if parsed.get("cuisine") else None
    parsed["health_goal"] = normalize_text(parsed["health_goal"]) if parsed.get("health_goal") else None
    parsed["style_keywords"] = [
        normalize_text(x) for x in parsed.get("style_keywords", []) if normalize_text(x)
    ]
    parsed["semantic_query"] = parsed.get("semantic_query", user_query)
    parsed["keyword_query"] = normalize_text(parsed.get("keyword_query", user_query))

    if parsed.get("max_time") is not None:
        try:
            parsed["max_time"] = int(parsed["max_time"])
        except Exception:
            parsed["max_time"] = None

    return parsed


# =========================
# Retrieval
# =========================
def retrieve_semantic_candidates(parsed_query, candidate_k=30):
    """
    Pure semantic retrieval first. No hard filters yet.
    Faster and avoids zero-result issues caused by strict filtering.
    """
    df = load_data(DATA_PATH)
    model = get_embedding_model()
    recipe_embeddings = get_recipe_embeddings()

    semantic_query = parsed_query.get("semantic_query", "").strip()
    if not semantic_query:
        semantic_query = parsed_query.get("keyword_query", "").strip()

    query_embedding = model.encode(
        [semantic_query],
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # cosine similarity = dot product because vectors are normalized
    scores = np.dot(recipe_embeddings, query_embedding[0])

    # faster than full sort
    candidate_k = min(candidate_k, len(df))
    top_idx = np.argpartition(-scores, candidate_k - 1)[:candidate_k]
    top_scores = scores[top_idx]

    candidates = df.iloc[top_idx].copy()
    candidates["semantic_score"] = top_scores
    candidates = candidates.sort_values("semantic_score", ascending=False).copy()

    return candidates


# =========================
# Hard Constraints (minimal only)
# =========================

def apply_minimal_filters(candidates, parsed_query):
    data = candidates.copy()

    exclude_ingredients = parsed_query.get("exclude_ingredients", [])
    for ing in exclude_ingredients:
        data = data[~data["ingredients_text_norm"].apply(lambda x: ing in x)]

    
    dietary_constraints = parsed_query.get("dietary_constraints", [])
    for constraint in dietary_constraints:
        col = f"is_{constraint}"
        if col in data.columns:
            data = data[data[col] == True]

    return data.copy()


# =========================
# Soft Reranking
# =========================

def compute_rerank_scores(candidates, parsed_query):
    data = candidates.copy()
    data["rerank_score"] = data["semantic_score"]

    include_ingredients = parsed_query.get("include_ingredients", [])
    keyword_terms = [t for t in parsed_query.get("keyword_query", "").split() if t]

    # -------------------------
    # 1. Ingredient bonus
    # -------------------------
    if include_ingredients:
        for ing in include_ingredients:
            # ingredients 
            data.loc[
                data["ingredients_text_norm"].apply(lambda x: ing in x),
                "rerank_score"
            ] += 0.35

            data.loc[
                data["recipe_title_norm"].apply(lambda x: ing in x),
                "rerank_score"
            ] += 0.18

    # -------------------------
    # 2. Title keyword overlap bonus
    # -------------------------
    data["title_bonus"] = data["recipe_title_norm"].apply(
        lambda x: keyword_overlap_score(x, keyword_terms)
    )
    data["rerank_score"] += 0.20 * data["title_bonus"]

    # -------------------------
    # 3. Course bonus (soft)
    # -------------------------
    course = parsed_query.get("course")
    if course:
        aliases = course_aliases(course)
        data.loc[data["course_primary_norm"].isin(aliases), "rerank_score"] += 0.22

    # -------------------------
    # 4. Cuisine bonus
    # -------------------------
    cuisine = parsed_query.get("cuisine")
    if cuisine:
        data.loc[data["cuisine_primary_norm"] == cuisine, "rerank_score"] += 0.12

    # -------------------------
    # 5. Time scoring 
    # -------------------------
    max_time = parsed_query.get("max_time")
    # if max_time is not None:
        
    #     data.loc[data["total_time_min"] <= max_time, "rerank_score"] += 0.35

    #     data.loc[
    #         (data["total_time_min"] > max_time) & (data["total_time_min"] <= max_time + 15),
    #         "rerank_score"
    #     ] -= 0.03

    #     data.loc[
    #         (data["total_time_min"] > max_time + 15) & (data["total_time_min"] <= max_time + 30),
    #         "rerank_score"
    #     ] -= 0.10

    #     data.loc[
    #         data["total_time_min"] > max_time + 30,
    #         "rerank_score"
    #     ] -= 0.30
    if max_time is not None:
        data.loc[data["total_time_min"] <= max_time, "rerank_score"] += 0.40

        data.loc[
            (data["total_time_min"] > max_time) & (data["total_time_min"] <= max_time + 10),
            "rerank_score"
        ] -= 0.05

        data.loc[
            (data["total_time_min"] > max_time + 10) & (data["total_time_min"] <= max_time + 20),
            "rerank_score"
        ] -= 0.15

        data.loc[
            data["total_time_min"] > max_time + 20,
            "rerank_score"
        ] -= 0.35

    # -------------------------
    # 6. Health scoring 
    # -------------------------
    health_goal = parsed_query.get("health_goal")
    if health_goal:
        health_goal = normalize_text(health_goal)

        if health_goal in ["healthy", "low_calorie", "low calorie", "light"]:
            data.loc[
                data["health_level_norm"].isin(["healthy", "very healthy"]),
                "rerank_score"
            ] += 0.25

            
            data["rerank_score"] += 0.15 * data["healthiness_score"].fillna(0) / 100.0

            
            data.loc[
                data["health_level_norm"].isin(["unhealthy", "less healthy"]),
                "rerank_score"
            ] -= 0.10

    # -------------------------
    # 7. Quick keyword bonus
    # -------------------------
    style_keywords = parsed_query.get("style_keywords", [])
    if "quick" in style_keywords:
        data.loc[data["total_time_min"] <= 30, "rerank_score"] += 0.25
        data.loc[
            (data["total_time_min"] > 30) & (data["total_time_min"] <= 45),
            "rerank_score"
        ] += 0.08
        data.loc[data["total_time_min"] > 60, "rerank_score"] -= 0.12

    # -------------------------
    # 8. Main ingredient bonus
    # -------------------------
    for ing in include_ingredients:
        if ing == "chicken":
            data.loc[data["main_ingredient_norm"] == "poultry", "rerank_score"] += 0.18
        elif ing == "beef":
            data.loc[data["main_ingredient_norm"] == "red meat", "rerank_score"] += 0.18
        elif ing in ["fish", "salmon", "shrimp", "seafood"]:
            data.loc[data["main_ingredient_norm"].isin(["seafood", "fish"]), "rerank_score"] += 0.18

    # -------------------------
    # 9. Dinner/main preference
    # -------------------------
        # data.loc[data["course_primary_norm"] == "main", "rerank_score"] += 0.12
        # data.loc[data["course_primary_norm"].isin(["drink", "dessert"]), "rerank_score"] -= 0.20
    if parsed_query.get("course") == "dinner":
        data.loc[data["course_primary_norm"] == "main", "rerank_score"] += 0.18
        data.loc[data["course_primary_norm"].isin(["side", "soup"]), "rerank_score"] -= 0.10
        data.loc[data["course_primary_norm"].isin(["drink", "dessert"]), "rerank_score"] -= 0.25

    # -------------------------
    # 10. punish
    # -------------------------
    data.loc[data["total_time_min"] > 180, "rerank_score"] -= 0.15

    data = data.sort_values("rerank_score", ascending=False).copy()
    return data

# =========================
# Fallback Strategy
# =========================
def fallback_retrieval(parsed_query, candidate_k=30):
    """
    Progressive fallback when constraints are too strict.
    """
    # Round 1: semantic + minimal filters + rerank
    candidates = retrieve_semantic_candidates(parsed_query, candidate_k=candidate_k)
    candidates = apply_minimal_filters(candidates, parsed_query)
    ranked = compute_rerank_scores(candidates, parsed_query)

    if not ranked.empty:
        return ranked

    # Round 2: remove course / time pressure from scoring logic by relaxing query fields
    relaxed = dict(parsed_query)
    relaxed["course"] = None
    relaxed["max_time"] = None

    candidates = retrieve_semantic_candidates(relaxed, candidate_k=candidate_k)
    candidates = apply_minimal_filters(candidates, relaxed)
    ranked = compute_rerank_scores(candidates, relaxed)

    if not ranked.empty:
        return ranked

    # Round 3: full semantic only
    candidates = retrieve_semantic_candidates(
        {
            "semantic_query": parsed_query.get("semantic_query", ""),
            "keyword_query": parsed_query.get("keyword_query", ""),
            "include_ingredients": [],
            "exclude_ingredients": [],
            "dietary_constraints": [],
            "course": None,
            "cuisine": None,
            "max_time": None,
            "health_goal": None,
            "style_keywords": []
        },
        candidate_k=candidate_k
    )
    ranked = compute_rerank_scores(candidates, {
        "semantic_query": parsed_query.get("semantic_query", ""),
        "keyword_query": parsed_query.get("keyword_query", ""),
        "include_ingredients": [],
        "exclude_ingredients": [],
        "dietary_constraints": [],
        "course": None,
        "cuisine": None,
        "max_time": None,
        "health_goal": None,
        "style_keywords": []
    })

    return ranked


# =========================
# Explanation
# =========================
EXPLANATION_SYSTEM_PROMPT = """
You are a helpful recipe recommendation assistant.
Briefly explain which recipes best match the user's request and why.
Keep the explanation concise and practical.
""".strip()


def explain_results(user_query, retrieved_df, model="gpt-4o-mini"):
    if retrieved_df.empty:
        return "No suitable recipes were found."

    context_text = "\n\n".join(
        [build_recipe_context(row) for _, row in retrieved_df.head(3).iterrows()]
    )

    user_prompt = f"""
User request:
{user_query}

Retrieved recipes:
{context_text}

Please explain briefly which retrieved recipes best match the request.
Prioritize recipes that better satisfy time, health, and meal-type requirements.
If some retrieved results do not fully satisfy the request, mention that clearly.
Use concise bullet points.
""".strip()

    messages = [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    return call_llm(messages, model=model, temperature=0)


# =========================
# Main Pipeline
# =========================

def intelligent_recipe_search(
    user_query,
    top_k=5,
    candidate_k=50,
    model="gpt-4o-mini",
    use_explanation=True
):
    parsed_query = parse_nl_query(user_query, model=model)

    ranked = fallback_retrieval(parsed_query, candidate_k=candidate_k)

    if ranked.empty:
        return {
            "parsed_query": parsed_query,
            "results": pd.DataFrame(),
            "explanation": "No suitable recipes were found."
        }

    max_time = parsed_query.get("max_time")

    # top_k
    if max_time is not None:
        short_results = ranked[ranked["total_time_min"] <= max_time + 20].copy()

        if not short_results.empty:
            results = short_results.head(top_k).copy()
        else:
            results = ranked.head(top_k).copy()
    else:
        results = ranked.head(top_k).copy()

    if use_explanation:
        explanation = explain_results(user_query, results, model=model)
    else:
        explanation = ""

    return {
        "parsed_query": parsed_query,
        "results": results,
        "explanation": explanation
    }