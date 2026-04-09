import json
import ast
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/recipes_project_subset_4000.csv"


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


def normalize_list_values(values):
    return [normalize_text(v) for v in values if normalize_text(v) != ""]


def tokenize_query(text):
    tokens = normalize_text(text).split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens


def parse_comma_separated_input(text):
    if text is None:
        return []
    parts = [normalize_text(x) for x in str(text).split(",")]
    return [p for p in parts if p != ""]


def build_full_recipe_text(row):
    ingredients_text = ", ".join(row["ingredients"]) if isinstance(row["ingredients"], list) else ""
    directions_text = " ".join(row["directions"]) if isinstance(row["directions"], list) else ""
    cuisine_text = ", ".join(row["cuisine_list"]) if isinstance(row["cuisine_list"], list) else ""
    course_text = ", ".join(row["course_list"]) if isinstance(row["course_list"], list) else ""
    tastes_text = ", ".join(row["tastes"]) if isinstance(row["tastes"], list) else ""
    dietary_text = ", ".join(row["dietary_profile"]) if isinstance(row["dietary_profile"], list) else ""
    health_flags_text = ", ".join(row["health_flags"]) if isinstance(row["health_flags"], list) else ""

    text = f"""
Recipe Title: {row.get('recipe_title', '')}
Category: {row.get('category', '')}
Subcategory: {row.get('subcategory', '')}
Description: {row.get('description', '')}

Cuisine: {cuisine_text}
Course: {course_text}
Tastes: {tastes_text}
Primary Taste: {row.get('primary_taste', '')}
Secondary Taste: {row.get('secondary_taste', '')}

Main Ingredient: {row.get('main_ingredient', '')}
Difficulty: {row.get('difficulty', '')}
Cook Speed: {row.get('cook_speed', '')}
Health Level: {row.get('health_level', '')}
Healthiness Score: {row.get('healthiness_score', '')}

Dietary Profile: {dietary_text}
Health Flags: {health_flags_text}

Preparation Time (min): {row.get('est_prep_time_min', '')}
Cooking Time (min): {row.get('est_cook_time_min', '')}
Total Time (min): {row.get('total_time_min', '')}

Ingredients:
{ingredients_text}

Directions:
{directions_text}
""".strip()

    return text


def load_and_prepare_data(data_path=DATA_PATH):
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

    df["recipe_title_norm"] = df["recipe_title"].apply(normalize_text)
    df["search_text_norm"] = df["search_text"].apply(normalize_text)

    df["ingredients_norm"] = df["ingredients"].apply(normalize_list_values)
    df["ingredients_text_norm"] = df["ingredients_norm"].apply(lambda x: " | ".join(x))

    df["course_primary_norm"] = df["course_primary"].apply(normalize_text)
    df["cuisine_primary_norm"] = df["cuisine_primary"].apply(normalize_text)
    df["main_ingredient_norm"] = df["main_ingredient"].apply(normalize_text)

    df["metadata_text_norm"] = (
        df["category"].fillna("").astype(str).apply(normalize_text) + " " +
        df["subcategory"].fillna("").astype(str).apply(normalize_text) + " " +
        df["course_primary"].fillna("").astype(str).apply(normalize_text) + " " +
        df["cuisine_primary"].fillna("").astype(str).apply(normalize_text) + " " +
        df["primary_taste"].fillna("").astype(str).apply(normalize_text) + " " +
        df["secondary_taste"].fillna("").astype(str).apply(normalize_text) + " " +
        df["difficulty"].fillna("").astype(str).apply(normalize_text) + " " +
        df["health_level"].fillna("").astype(str).apply(normalize_text) + " " +
        df["main_ingredient"].fillna("").astype(str).apply(normalize_text)
    ).str.strip()

    df["full_info_text"] = df.apply(build_full_recipe_text, axis=1)

    return df


def build_index(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(df["search_text_norm"])
    return vectorizer, tfidf_matrix


def apply_structured_filters(
    data,
    include_ingredients=None,
    exclude_ingredients=None,
    max_time=None,
    course=None,
    cuisine=None,
    vegan=None,
    vegetarian=None,
    gluten_free=None,
    dairy_free=None,
    nut_free=None,
    halal=None,
    kosher=None
):
    mask = pd.Series(True, index=data.index)

    if max_time is not None and max_time > 0:
        mask &= data["total_time_min"] <= max_time

    if course is not None and str(course).strip() != "" and str(course).lower() != "any":
        course_norm = normalize_text(course)
        mask &= data["course_primary_norm"] == course_norm

    if cuisine is not None and str(cuisine).strip() != "" and str(cuisine).lower() != "any":
        cuisine_norm = normalize_text(cuisine)
        mask &= data["cuisine_primary_norm"] == cuisine_norm

    flag_map = {
        "is_vegan": vegan,
        "is_vegetarian": vegetarian,
        "is_gluten_free": gluten_free,
        "is_dairy_free": dairy_free,
        "is_nut_free": nut_free,
        "is_halal": halal,
        "is_kosher": kosher
    }

    for col, value in flag_map.items():
        if col in data.columns and value is True:
            mask &= data[col] == True

    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []

    include_ingredients = [normalize_text(x) for x in include_ingredients if normalize_text(x) != ""]
    exclude_ingredients = [normalize_text(x) for x in exclude_ingredients if normalize_text(x) != ""]

    for ing in include_ingredients:
        mask &= data["ingredients_text_norm"].apply(lambda x: ing in x)

    for ing in exclude_ingredients:
        mask &= ~data["ingredients_text_norm"].apply(lambda x: ing in x)

    return mask


def compute_title_bonus(title_text, query_terms):
    if len(query_terms) == 0:
        return 0.0
    hits = sum(1 for term in query_terms if term in title_text)
    return hits / len(query_terms)


def compute_metadata_bonus(metadata_text, query_terms):
    if len(query_terms) == 0:
        return 0.0
    hits = sum(1 for term in query_terms if term in metadata_text)
    return hits / len(query_terms)


def compute_include_ingredient_bonus(ingredients_text, include_ingredients):
    if include_ingredients is None or len(include_ingredients) == 0:
        return 0.0

    include_ingredients = [normalize_text(x) for x in include_ingredients if normalize_text(x) != ""]
    if len(include_ingredients) == 0:
        return 0.0

    hits = sum(1 for ing in include_ingredients if ing in ingredients_text)
    return hits / len(include_ingredients)


# 初始化：读取数据并建索引
df = load_and_prepare_data(DATA_PATH)
vectorizer, tfidf_matrix = build_index(df)


def classic_recipe_search(
    query,
    top_k=10,
    include_ingredients=None,
    exclude_ingredients=None,
    max_time=None,
    course=None,
    cuisine=None,
    vegan=None,
    vegetarian=None,
    gluten_free=None,
    dairy_free=None,
    nut_free=None,
    halal=None,
    kosher=None,
    min_tfidf_score=0.0
):
    query = str(query).strip()
    if query == "":
        raise ValueError("Query cannot be empty.")

    candidate_mask = apply_structured_filters(
        df,
        include_ingredients=include_ingredients,
        exclude_ingredients=exclude_ingredients,
        max_time=max_time,
        course=course,
        cuisine=cuisine,
        vegan=vegan,
        vegetarian=vegetarian,
        gluten_free=gluten_free,
        dairy_free=dairy_free,
        nut_free=nut_free,
        halal=halal,
        kosher=kosher
    )

    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    query_norm = normalize_text(query)
    query_terms = tokenize_query(query)

    query_vector = vectorizer.transform([query_norm])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix[candidate_indices]).flatten()

    results = df.iloc[candidate_indices].copy()
    results["tfidf_score"] = similarity_scores
    results = results[results["tfidf_score"] >= min_tfidf_score].copy()

    if len(results) == 0:
        return pd.DataFrame()

    include_ingredients = include_ingredients or []

    results["title_bonus"] = results["recipe_title_norm"].apply(
        lambda x: compute_title_bonus(x, query_terms)
    )

    results["metadata_bonus"] = results["metadata_text_norm"].apply(
        lambda x: compute_metadata_bonus(x, query_terms)
    )

    results["ingredient_bonus"] = results["ingredients_text_norm"].apply(
        lambda x: compute_include_ingredient_bonus(x, include_ingredients)
    )

    results["final_score"] = (
        results["tfidf_score"] +
        0.20 * results["title_bonus"] +
        0.10 * results["metadata_bonus"] +
        0.20 * results["ingredient_bonus"]
    )

    results = results.sort_values("final_score", ascending=False).copy()

    results["ingredients_preview"] = results["ingredients"].apply(
        lambda x: ", ".join(x[:5]) if isinstance(x, list) else ""
    )

    results["directions_preview"] = results["directions"].apply(
        lambda x: " ".join(x[:2]) if isinstance(x, list) else ""
    )

    results = results.head(top_k).copy()

    return results