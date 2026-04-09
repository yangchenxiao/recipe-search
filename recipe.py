# import streamlit as st
# from search_engine import classic_recipe_search
# from intelligent_search import intelligent_recipe_search

# st.set_page_config(
#     page_title="Recipe Search Demo",
#     page_icon="🍳",
#     layout="wide"
# )

# st.title("🍳 Recipe Search System")
# st.caption("Compare keyword-based retrieval with LLM-based intelligent search on recipe queries.")
# st.info("Try queries like: 'chicken', 'quick healthy chicken dinner', or 'chicken dinner under 30 minutes'")

# mode = st.sidebar.radio(
#     "Search Mode",
#     ["Classic IR", "Intelligent Search", "Compare"]
# )

# query = st.text_input("Enter your query")

# top_k = st.sidebar.slider("Top K", 1, 10, 5)

# if st.button("Search"):
#     if query.strip() == "":
#         st.warning("Please enter a query.")
#     else:
#         if mode == "Classic IR":
#             results = classic_recipe_search(query=query, top_k=top_k)

#             st.subheader("Classic IR Results")
#             if results.empty:
#                 st.info("No matching recipes found.")
#             else:
#                 for _, row in results.iterrows():
#                     st.markdown(f"### {row['recipe_title']}")
#                     st.write(
#                         f"Course: {row['course_primary']} | "
#                         f"Cuisine: {row['cuisine_primary']} | "
#                         f"Time: {row['total_time_min']} min"
#                     )
#                     st.write(f"Ingredients: {row['ingredients_preview']}")
#                     st.write("---")

#         elif mode == "Intelligent Search":
#             output = intelligent_recipe_search(user_query=query, top_k=top_k)

#             st.subheader("Parsed Query")
#             st.json(output["parsed_query"])

#             st.subheader("LLM Search Results")
#             if output["results"].empty:
#                 st.info("No matching recipes found.")
#             else:
#                 for _, row in output["results"].iterrows():
#                     st.markdown(f"### {row['recipe_title']}")
#                     st.write(
#                         f"Course: {row['course_primary']} | "
#                         f"Cuisine: {row['cuisine_primary']} | "
#                         f"Time: {row['total_time_min']} min"
#                     )
#                     st.write("---")

#             st.subheader("AI Explanation")
#             st.write(output["explanation"])

#         else:  # Compare
#             col1, col2 = st.columns(2)

#             classic_results = classic_recipe_search(query=query, top_k=top_k)
#             llm_output = intelligent_recipe_search(user_query=query, top_k=top_k)

#             with col1:
#                 st.subheader("Classic IR")
#                 if classic_results.empty:
#                     st.info("No results.")
#                 else:
#                     for _, row in classic_results.iterrows():
#                         st.markdown(f"**{row['recipe_title']}**")
#                         st.caption(
#                             f"{row['course_primary']} | {row['cuisine_primary']} | {row['total_time_min']} min"
#                         )

#             with col2:
#                 st.subheader("Intelligent Search")
#                 if llm_output["results"].empty:
#                     st.info("No results.")
#                 else:
#                     for _, row in llm_output["results"].iterrows():
#                         st.markdown(f"**{row['recipe_title']}**")
#                         st.caption(
#                             f"{row['course_primary']} | {row['cuisine_primary']} | {row['total_time_min']} min"
#                         )

#                 st.markdown("#### AI Explanation")
#                 st.write(llm_output["explanation"])

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
                    st.markdown(f"### {row['recipe_title']}")
                    st.write(
                        f"Course: {row['course_primary']} | "
                        f"Cuisine: {row['cuisine_primary']} | "
                        f"Time: {row['total_time_min']} min"
                    )
                    st.write(f"Ingredients: {row['ingredients_preview']}")
                    st.markdown("---")

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
                    st.markdown(f"### {row['recipe_title']}")

                    time_text = f"{row['total_time_min']} min"
                    if max_time is not None and pd.notna(row["total_time_min"]):
                        if row["total_time_min"] <= max_time:
                            time_text += " ✅ within limit"
                        else:
                            time_text += " ⚠ above limit"

                    st.write(
                        f"Course: {row['course_primary']} | "
                        f"Cuisine: {row['cuisine_primary']} | "
                        f"Time: {time_text}"
                    )
                    st.markdown("---")

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
                        st.markdown(f"**{row['recipe_title']}**")
                        st.caption(
                            f"{row['course_primary']} | {row['cuisine_primary']} | {row['total_time_min']} min"
                        )

            with col2:
                st.subheader("Intelligent Search")
                if llm_output["results"].empty:
                    st.info("No results.")
                else:
                    max_time = llm_output["parsed_query"].get("max_time")

                    for _, row in llm_output["results"].iterrows():
                        time_text = f"{row['total_time_min']} min"
                        if max_time is not None and pd.notna(row["total_time_min"]):
                            if row["total_time_min"] <= max_time:
                                time_text += " ✅"
                            else:
                                time_text += " ⚠"

                        st.markdown(f"**{row['recipe_title']}**")
                        st.caption(
                            f"{row['course_primary']} | {row['cuisine_primary']} | {time_text}"
                        )

                st.markdown("#### AI Explanation")
                st.write(llm_output["explanation"])