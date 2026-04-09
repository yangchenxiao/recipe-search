import streamlit as st
import pandas as pd
from search_engine import classic_recipe_search
from intelligent_search import intelligent_recipe_search

st.set_page_config(
    page_title="Recipe Search System",
    page_icon="🍳",
    layout="wide"
)

st.title("🍳 Recipe Search System")
st.caption("Compare keyword-based retrieval with LLM-based intelligent search.")
st.info("Example queries: chicken | quick healthy chicken dinner | chicken dinner under 30 minutes")

mode = st.sidebar.radio(
    "Search Mode",
    ["Classic IR", "Intelligent Search", "Compare"]
)

top_k = st.sidebar.slider("Top K", 1, 10, 5)

query = st.text_input("Enter your query")


def get_display_value(row, possible_cols):
    """
    Return the first usable value from a list of possible column names.
    Safe for strings, lists, tuples, and missing values.
    """
    for col in possible_cols:
        if col not in row.index:
            continue

        value = row[col]

        if value is None:
            continue

        if isinstance(value, float) and pd.isna(value):
            continue

        if isinstance(value, str):
            if value.strip() == "":
                continue
            return value

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            return value

        return value

    return None


def format_value(value, sep=", "):
    """
    Format list/tuple values nicely for display; otherwise convert to string.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return sep.join(str(x) for x in value)
    return str(value)


def render_full_recipe(row, show_time_flag=False, max_time=None):
    st.markdown(f"### {row['recipe_title']}")

    time_text = f"{row['total_time_min']} min"
    if show_time_flag and max_time is not None and pd.notna(row["total_time_min"]):
        if row["total_time_min"] <= max_time:
            time_text += " ✅ within limit"
        else:
            time_text += " ⚠ above limit"

    st.write(
        f"Course: {row['course_primary']} | "
        f"Cuisine: {row['cuisine_primary']} | "
        f"Time: {time_text}"
    )

    ingredients = get_display_value(
        row,
        ["ingredients_preview", "ingredients", "ingredient_list", "ingredients_full"]
    )
    if ingredients is not None:
        st.write(f"Ingredients: {format_value(ingredients)}")

    components = get_display_value(
        row,
        ["components", "recipe_components"]
    )
    if components is not None:
        st.write(f"Components: {format_value(components)}")

    directions = get_display_value(
        row,
        ["directions", "instructions", "steps", "method", "recipe_instructions"]
    )
    if directions is not None:
        st.write(f"Directions: {format_value(directions, sep=' ')}")

    st.markdown("---")


if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        if mode == "Classic IR":
            results = classic_recipe_search(query=query, top_k=top_k)

            st.subheader("Classic IR Results")
            if results.empty:
                st.info("No matching recipes found.")
            else:
                for _, row in results.iterrows():
                    render_full_recipe(row)

        elif mode == "Intelligent Search":
            output = intelligent_recipe_search(user_query=query, top_k=top_k)

            with st.expander("Parsed Query (click to expand)"):
                st.json(output["parsed_query"])

            st.subheader("LLM Search Results")
            if output["results"].empty:
                st.info("No matching recipes found.")
            else:
                max_time = output["parsed_query"].get("max_time")

                for _, row in output["results"].iterrows():
                    render_full_recipe(row, show_time_flag=True, max_time=max_time)

            st.subheader("AI Explanation")
            st.write(output["explanation"])

        else:  # Compare
            st.info("Left: Classic IR (keyword-based). Right: Intelligent Search (LLM + semantic retrieval + explanation).")

            classic_results = classic_recipe_search(query=query, top_k=top_k)
            llm_output = intelligent_recipe_search(user_query=query, top_k=top_k)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classic IR")
                if classic_results.empty:
                    st.info("No results.")
                else:
                    for _, row in classic_results.iterrows():
                        render_full_recipe(row)

            with col2:
                st.subheader("Intelligent Search")
                if llm_output["results"].empty:
                    st.info("No results.")
                else:
                    max_time = llm_output["parsed_query"].get("max_time")

                    for _, row in llm_output["results"].iterrows():
                        render_full_recipe(row, show_time_flag=True, max_time=max_time)

                st.markdown("#### AI Explanation")
                st.write(llm_output["explanation"])