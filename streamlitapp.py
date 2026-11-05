
import os
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "https://industry-search-ranking-system-prince.onrender.com"
).rstrip("/")

PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="SearchRank | Intelligent Product Search",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "search_history" not in st.session_state:
    st.session_state.search_history = []

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    padding: 32px 0 25px 0;
}

.brand {
    color: #60a5fa;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    color: #f8fafc;
    margin: 12px 0;
}

.hero p {
    color: #94a3b8;
    font-size: 15px;
}

div[data-testid="stTextInput"] input {
    background: #151e30;
    color: white;
    border: 1px solid #334155;
    border-radius: 10px;
    min-height: 48px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1d4ed8;
    color: white;
}

.product-card {
    background: #151e30;
    border: 1px solid #263449;
    border-radius: 15px;
    padding: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 30px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


def format_price(value):
    try:
        return f"₹{float(value):,.2f}"
    except (ValueError, TypeError):
        return "Price unavailable"


def format_rating(value):
    try:
        return f"⭐ {float(value):.1f}/5"
    except (ValueError, TypeError):
        return "Rating unavailable"


def format_score(value):
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return "Unavailable"


def search_products(query):
    try:
        response = requests.post(
            PREDICT_URL,
            json={"query": query},
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return None, "Unexpected API response format."

        return data, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the FastAPI backend."

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text

        return None, f"API Error: {detail}"

    except (ValueError, requests.exceptions.RequestException) as error:
        return None, f"Search failed: {error}"


st.markdown("""
<div class="hero">
    <div class="brand">SEARCHRANK / INTELLIGENT SEARCH</div>
    <h1>Find Products.<br>Discover Better Results.</h1>
    <p>
        An intelligent product search engine powered by
        TF-IDF retrieval and XGBoost Learning-to-Rank.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔍 Search Products")

st.caption("Enter a product name, category, or keyword.")

with st.form("product_search_form"):
    query = st.text_input(
        "Product query",
        placeholder="Try: Samsung monitor, gaming laptop...",
        label_visibility="collapsed"
    )

    search_clicked = st.form_submit_button(
        "Search Products →",
        use_container_width=True
    )

if search_clicked:
    if not query.strip():
        st.warning("Please enter a product search query.")

    else:
        with st.spinner("Finding and ranking the best products..."):
            data, error = search_products(query.strip())

        if error:
            st.error(error)
            st.session_state.search_results = None

        else:
            st.session_state.search_results = data
            st.session_state.last_query = query.strip()

            st.session_state.search_history.insert(
                0, query.strip()
            )

            st.session_state.search_history = (
                st.session_state.search_history[:5]
            )

results = st.session_state.search_results

if results is not None:

    st.markdown("---")
    st.markdown("## Search Results")

    st.caption(
        f"Results for: {st.session_state.last_query}"
    )

    if not results:
        st.info("No matching products found.")

    else:
        st.success(f"{len(results)} products found")

        st.caption(
            "Products are displayed in the order returned "
            "by the XGBoost ranking API."
        )

        for index, product in enumerate(results, start=1):

            title = product.get(
                "product_title", "Unknown Product"
            )

            brand = product.get("brand", "Unknown")

            price = format_price(
                product.get("price")
            )

            rating = format_rating(
                product.get("rating")
            )

            score = format_score(
                product.get("ranking_score")
            )

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.caption(f"🏆 #{index} RANKED PRODUCT")
                    st.subheader(str(title))
                    st.caption(f"Brand: {brand}")

                with col2:
                    st.metric("Rank", f"#{index}")

                price_col, rating_col, score_col = st.columns(3)

                with price_col:
                    st.metric("Price", price)

                with rating_col:
                    st.metric("Rating", rating)

                with score_col:
                    st.metric("XGBoost Score", score)

        st.caption(
            "Ranking scores are raw model outputs, not percentages."
        )

if st.session_state.search_history:

    with st.expander("🕘 Recent Searches"):
        for previous_query in st.session_state.search_history:
            st.write(f"• {previous_query}")

st.markdown("""
<div class="footer">
    SEARCHRANK · Powered by TF-IDF + XGBoost Learning-to-Rank
    <br><br>
    Built with Streamlit and FastAPI
</div>
""", unsafe_allow_html=True)
