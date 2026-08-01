import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Search-Ranking-System",
    page_icon="🛒",
    layout="wide"
)

st.title("Industry Search Ranking System")

st.write("Search products ranked using XGBoost Learning-to-Rank")

query = st.text_input(
    "Enter Product Query",
    placeholder="gaming laptop"
)

if st.button("search"):
    if query.strip=="":
        st.warning("please enter a search query")

    else:
        response = requests.post(

            "http://127.0.0.1:8000/predict",

            json={"query": query}

        )
        if response.status_code==200:
            data=response.json()
            df=pd.DataFrame(data)
            st.success(f"{len(df)} products found")
            for i, row in df.iterrows():

                st.container()

                col1, col2 = st.columns([4,1])

                with col1:

                    st.subheader(row["product_title"])

                    st.write(f"**Brand:** {row['brand']}")

                    st.write(f"⭐ Rating: {row['rating']}")

                with col2:

                    st.metric(
                        "Price",
                        f"₹{row['price']:,}"
                    )

                    st.metric(
                        "Score",
                        round(row["ranking_score"],2)
                    )

                st.divider()
        else:
            st.error("prediction failed.")