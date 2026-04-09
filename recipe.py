
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


def normalize_to_list(value):
    """
    Convert a value into a lowercase string list for matching.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(x).strip().lower() for x in value if str(x).strip() != ""]

    if isinstance(value, str):
        parts = [x.strip().lower() for x in value.split(",") if x.strip() != ""]
        return parts if parts else [value.strip().lower()]

    return [str(value).strip().lower()]


def render_match_summary(row, parsed_query):
    """
    Show matched / unmatched conditions for Intelligent Search results.
    Only uses fields if they exist.
    """
    if not parsed_query:
        return

    matched = []
    unmatched = []

    include_ingredients = parsed_query.get("include_ingredients")
    exclude_ingredients = parsed_query.get("exclude_ingredients")
    max_time = parsed_query.get("max_time")
    course = parsed_query.get("course")
    cuisine = parsed_query.get("cuisine")
    dietary_constraints = parsed_query.get("dietary_constraints")
    health_goal = parsed_query.get("health_goal")
    style_keywords = parsed_query.get("style_keywords")

    recipe_ingredients = get_display_value(
        row,
        ["ingredients", "ingredient_list", "ingredients_full", "ingredients_preview"]
    )
    recipe_ingredients_text = format_value(recipe_ingredients, sep=", ")
    recipe_ingredients_lower = recipe_ingredients_text.lower() if recipe_ingredients_text else ""

    recipe_course = str(get_display_value(row, ["course_primary", "course"])).lower() if get_display_value(row, ["course_primary", "course"]) is not None else ""
    recipe_cuisine = str(get_display_value(row, ["cuisine_primary", "cuisine"])).lower() if get_display_value(row, ["cuisine_primary", "cuisine"]) is not None else ""
    recipe_time = get_display_value(row, ["total_time_min"])

    recipe_tags = normalize_to_list(
        get_display_value(
            row,
            ["nutrition_health_tags", "health_tags", "dietary_tags", "tags", "nutrition_tags"]
        )
    )

    if include_ingredients:
        for ing in include_ingredients:
            ing_lower = str(ing).strip().lower()
            if ing_lower and ing_lower in recipe_ingredients_lower:
                matched.append(f"contains {ing}")
            else:
                unmatched.append(f"missing {ing}")

    if exclude_ingredients:
        for ing in exclude_ingredients:
            ing_lower = str(ing).strip().lower()
            if ing_lower and ing_lower in recipe_ingredients_lower:
                unmatched.append(f"contains excluded ingredient: {ing}")
            else:
                matched.append(f"does not include {ing}")

    if max_time is not None and recipe_time is not None and pd.notna(recipe_time):
        if recipe_time <= max_time:
            matched.append(f"within {max_time} min")
        else:
            unmatched.append(f"over {max_time} min")

    if course:
        if str(course).strip().lower() in recipe_course:
            matched.append(f"course = {course}")
        else:
            unmatched.append(f"course not matched: {course}")

    if cuisine:
        if str(cuisine).strip().lower() in recipe_cuisine:
            matched.append(f"cuisine = {cuisine}")
        else:
            unmatched.append(f"cuisine not matched: {cuisine}")

    if dietary_constraints:
        for tag in dietary_constraints:
            tag_lower = str(tag).strip().lower()
            if tag_lower in recipe_tags:
                matched.append(f"dietary: {tag}")
            else:
                unmatched.append(f"dietary not matched: {tag}")

    if health_goal:
        health_lower = str(health_goal).strip().lower()
        if health_lower in recipe_tags:
            matched.append(f"health goal: {health_goal}")
        else:
            unmatched.append(f"health goal not matched: {health_goal}")

    if style_keywords:
        for keyword in style_keywords:
            keyword_lower = str(keyword).strip().lower()
            title_text = str(get_display_value(row, ["recipe_title", "title"])).lower() if get_display_value(row, ["recipe_title", "title"]) is not None else ""
            if keyword_lower in recipe_ingredients_lower or keyword_lower in title_text:
                matched.append(f"style keyword: {keyword}")

    if matched or unmatched:
        st.markdown("**Match Summary**")
        if matched:
            st.write("✅ Matched: " + "; ".join(matched))
        if unmatched:
            st.write("❌ Not matched: " + "; ".join(unmatched))


def render_parsed_query_summary(parsed_query):
    """
    Show a compact parsed query summary for Intelligent Search.
    """
    if not parsed_query:
        return

    summary_parts = []

    if parsed_query.get("include_ingredients"):
        summary_parts.append(f"Include: {', '.join(map(str, parsed_query['include_ingredients']))}")

    if parsed_query.get("exclude_ingredients"):
        summary_parts.append(f"Exclude: {', '.join(map(str, parsed_query['exclude_ingredients']))}")

    if parsed_query.get("max_time") is not None:
        summary_parts.append(f"Max time: {parsed_query['max_time']} min")

    if parsed_query.get("course"):
        summary_parts.append(f"Course: {parsed_query['course']}")

    if parsed_query.get("cuisine"):
        summary_parts.append(f"Cuisine: {parsed_query['cuisine']}")

    if parsed_query.get("dietary_constraints"):
        summary_parts.append(f"Dietary: {', '.join(map(str, parsed_query['dietary_constraints']))}")

    if parsed_query.get("health_goal"):
        summary_parts.append(f"Health goal: {parsed_query['health_goal']}")

    if parsed_query.get("style_keywords"):
        summary_parts.append(f"Style: {', '.join(map(str, parsed_query['style_keywords']))}")

    if summary_parts:
        st.markdown("**Parsed Query Summary**")
        for item in summary_parts:
            st.write(f"- {item}")


def render_full_recipe(row, show_time_flag=False, max_time=None, parsed_query=None):
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

    nutrition_or_health = get_display_value(
        row,
        ["nutrition_health_tags", "health_tags", "dietary_tags", "tags", "nutrition_tags"]
    )
    if nutrition_or_health is not None:
        st.write(f"Nutrition / Health Tags: {format_value(nutrition_or_health)}")

    recipe_id = get_display_value(
        row,
        ["recipe_id", "id", "recipe_code"]
    )
    if recipe_id is not None:
        st.write(f"Recipe ID: {format_value(recipe_id)}")

    source_link = get_display_value(
        row,
        ["source_url", "recipe_url", "url", "source_link"]
    )
    if source_link is not None:
        source_link_text = format_value(source_link)
        st.markdown(f"Source Link: [Open Recipe]({source_link_text})")

    if parsed_query is not None:
        render_match_summary(row, parsed_query)

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

            render_parsed_query_summary(output["parsed_query"])

            st.subheader("LLM Search Results")
            if output["results"].empty:
                st.info("No matching recipes found.")
            else:
                max_time = output["parsed_query"].get("max_time")

                for _, row in output["results"].iterrows():
                    render_full_recipe(
                        row,
                        show_time_flag=True,
                        max_time=max_time,
                        parsed_query=output["parsed_query"]
                    )

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
                render_parsed_query_summary(llm_output["parsed_query"])

                if llm_output["results"].empty:
                    st.info("No results.")
                else:
                    max_time = llm_output["parsed_query"].get("max_time")

                    for _, row in llm_output["results"].iterrows():
                        render_full_recipe(
                            row,
                            show_time_flag=True,
                            max_time=max_time,
                            parsed_query=llm_output["parsed_query"]
                        )

                st.markdown("#### AI Explanation")
                st.write(llm_output["explanation"])