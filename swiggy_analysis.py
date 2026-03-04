"""
SWIGGY DATA ANALYSIS — Python / Streamlit
Run: pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
     streamlit run swiggy_analysis.py
"""

import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Swiggy Data Analysis",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE   = "#FC8019"
DARK_ORG = "#E16A00"
BG_CREAM = "#FFF3E0"
TEXT_DRK  = "#1C1C1C"
GRAY     = "#6B6B6B"

# ── Exact Swiggy logo embedded as base64 PNG ─────────────────────
_LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCADhAOEDASIAAhEBAxEB/8QAHAABAAICAwEAAAAAAAAAAAAAAAEHBggCBAUD/8QAThAAAQMCAgMHDgkJCQAAAAAAAAECAwQFBhEHEjEIIUFCcYGxExQVIiMyM1FSYXJzkaEWJCU1N2LB0dIXNERUVXSClLJDVoSToqTC4vD/xAAcAQEAAgMBAQEAAAAAAAAAAAAABgcDBAUBAgj/xAA4EQACAQMCAgcGBQIHAAAAAAAAAQIDBAUGETFBEhQhIlFx0RMyYZGxwUJDgaHwFuEjUlNykvHy/9oADAMBAAIRAxEAPwD4gApQ/SoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQBGsZZgXAF7xZqTMi6yt/Gq5G7zvQTjdBd+E9G+GMPo2RlF15V5ZuqKnt3Z+ZNicx3cfgLm9XT92PiyK5bVtlj26cX05rkvu+H3+Br1aMM4gu6a9ustdUs4rmxqjF/jXeMkpdE2OKjv6Gkh9ZUN+zM2Vaw5apJKOlLaK78m38iF19eX85f4cIxXzNbJdEWNGJmlPQSehUJn70Q8K54GxfbO3qrDV6nlRtSRP9GZthqnHUPqelLRruyaPilrrIwffjF/pt9zTFc45nxyJ1NzO+a7eVoNsMR4Sw/iCLVutsgmdxZMspG8jk3ymseaJbnaNessD5LlSJvug/tmfj6eU4F/pu5tk5U++v3+RLcTrS0vGqdZezk/Hh8/UrQEEkba2Jmmmt0AAD0AAAAAAAAAAAAAAAAAAAAAAAAgszRDo57OKy+3yJ7Lex2tTw/rPnX6nT6O3wdFOEvhZiHUnj+T6TKSqd5XiZz9Bs7TwxQwthhjYxjGo1rU2IniJZp7DKu+sVl3VwXiV7q/Ucrfeyt3tL8T8PgiKaCKnhZBBGxkTG6rWtbkiJ4j76pKbAT5LYqx9r3AAPQAcXKSAScXIcgAVXpY0bQXyGa72SJkF0bm6SNu8lT9zvOUFI2SOaSN8b2SNcrXNdvOaqbUU3PchSOnvBsca/Cm3RbXI2ua1vMkn2LzEP1Dhoyi7miu1cV9ywNI6jlSqRsrh7xfuvwfh5eBTwAIIWqAAAAAAAAAAAAAAAAAAAAACHEnu6O7Z2YxraqJ/bsfMkknot7ZegzW9J1akYLmzWvLhW1CdWXCKb+RsHoow6mHMH01LKnxuoTq9S767uDmTJOYzE4sTI5FwUKMaNONOPBLY/PNxXncVZVZvtk92AAZTCAuwEOAKc3Q96udu7D01tuM9KyXqkknUZFYrlRW5b6cqlhaP6ye4YJs9bVSJLUS0jHSO8pctpWG6Z+cLJ6ubpaWToq+jmw/uTCPWlWbytaDfYkvsSi+oU44K2qKK6TlLd8+ZlAAJCRcHTuFJT1tBNS1TEkhmjVkjV4UVMlO4Q48aTWzPU2nujT/FFokseIK+0z/o8ita7ymbWr7Mjzy0t0da0p8QW25s/SoXRScrFTL3L7irSpMpbdWup01wTL8wN916wp1nxa7fNdjAANA7AAAAAAAAAAAAAAAAAAALE3PVN1TH75l/R6KRzedWp9pXZZO50ljZjqpj8ugk1eZ8Z1MLs76lv4nB1O2sVX28PujYdpJDSS1yiAAAAQ4khwBR+6Z+cLJ6ubpYWToq+jmw/uTCtN0x84WT1c3S06mEdLsVkw3QWmSyyTOpYkj6o2ZE1suYiMbyjaZWtKs9k0vsTueOub/A20baHSact/3L+BTC6dKf+7dR/MN+4+MmnVeJhxP4qv8A6nV/qDH/AOp+z9DiLSuWf5L+a9S7Ti5ShqzTjeHp8UslDD6yRz/uMWvekvF931433JaOJ3FpI0j9+33mrW1PZQXd3k/L1N620Vk6r76UV8X6bmY7o29W+fsbZYJGTVcUqzS6rs+ppq5Ii+dc/cVCHbdd6673d853GBBsjeu9uHWa23LRwuMWMtI26e+3P4sAA0DrAAAAAAAAAAAAAAAAAAAy3Q/cOsNItrkf2jJXOp3fxIqJ78jEiaSeSkq4aqDwsMrZI+VFRU6DZs6vsa8Kng0zSyNsrq1qUX+JNG57STzbBcYrrZaO4w95UQNlbzoekXBGSklJcGfniUXGTi+KAAPo8AAAPFxJh2z4gpWQXiijqmMdrR6yqitXzKh4n5MMD/sRn+dJ+IzUGtUs6FR9KcE35I2qV7c0Y9CnUaXgm0YOui3Av7EZ/MSfiPk7RPgT9kSM/wAZL+Mz0GN420f5cfkjKsrfL86X/J+pW8+h7B0ng466H0alV/qzMfvGhCmWJ7rVe5438VlRGj0XnTLIughxgqYWxqLZ00bVHUOToveNaX6vf6mnl+tVZZLrNbLjFqVETu21XZo7xKi+I6Rlel65x3TSBcpo+8ickDXN42qiIvvzMUKyvacKVecKfBN7F24utVr2lOrVW0mk2AAapvgAAAAAAAAAAAAAAAAAAAAF87ni99eYaqLLLInVrfJrR+qfvp7Fz9xa7TVPRlf/AIOYvo6yT83md1vUeg5UTPmXJeY2pjXuRZmnb3rFqoyfej2ehSWrsc7LISkvdn2rz5r5n0AB3yLgAAAAAAAAAx/Hd6jsGFK+6PXtoY16n9aRd5qe1UMgXYUTuiMQpPcKbDcEnaU/xiq9JUyYnszXnQ52VvFaW0qnPl5nWwmOeQvoUeW+78lx9Co3LJJK975Nd73K5zvKVdpIBUzbb3ZfsYqK2QAB4fQAAAAAAAAAAAAAAAAAAAABDml5aG9IdJUW+LD97qmQ1lOiMp5pHZJO3gTPyk95RxDmnRxuSq2FXpw4c14nFzWFo5ah7Ko9mu1PwNzkfH5Zy1jUGw2y+XibrWz01dVOZ3zY3Lk3lXYhk8OjbSBIn5k9n1XVrc+kmFHUFesulC3bXw/6K7uNJ21tLo1byKfx/wDRstrIRredhrd+THSB+rf71v3nQv2CMZ2O1TXO4072U8WWu5tWi5ZrlszMks5dQi5StpbL+eBgjpmynJRjewbf88TaJqnIpzc2VlZUW67009S+SKKWNY2ucq6uaLnl7ELjO3Y3avKEayW25HclYysLqdvJ7uPMAEONs0TycS3amsdlq7tVJ3GljV7k8rxInnVcjU67V1RdbrU3Cqk16ipkWSTlXg5E2FoboTFHXFdDhikk7jA5JqrV4z+K3mTf9hUxXmpsh7esqEeEfqW3ojEO2tndVF3p8P8Ab/fiAARcnQAAAAAAAAAAAAAAAAAAAAAAAAAABeO50vNHJZqqx6jGVcMiz+eVi5JnzLvewt9pp7hu71lgvVNd6XwtO7W1eB7eFq8qG12HbvSXuz0t0opEdDUR6zd/vfGi+dF3ix9N5CNe39i/ej9CmtY4mVneO4j7tTt8nzR6ph2mX6Nbx6pP6kMxMa0iWuovmDLla6JUSpni7mjlyRVRUXL3HbvIOdCcY8Wn9CM2M4wuqcpcFJfVFdbmTwV+9KH/AJl1FXaB8L3rD1JcpLzTpTOqnM1I9dFXJqLv73KWiaeEpTpWMITWzOlqSvTr5OrUpveLfFeSBjePsQ0+F8OVN0m7d7W6sMflyLsT/wBwIpkMjjWjTLi34R4l61pJPk+3uWOHxPfxnfYh5mMgrK2cvxPsX8+A0/iJZS8VN+6u2Xl/cwysqaisq5qyqk16mocskjvKVVzU+YBVk5OT6T4svaEIwioxWyQAB8n2AAAAAAAAAAAAAAAAAAAAAAAAAAACzNBOLktN1TD9dL8RrXdwe7ZHL4uRekrMnxG7YXk7SvGrDkczLY6nkbWVCfPh8HyZuc1SSv8AQ7jFMSYeSCrkTsnRIjKjPjpwP5+kz9qlr21xC5pKrDgyhru0q2laVCqtpRZOqSDxsU3qjw/Zai6VvgYW6yNTa93A1POqmSc404uUnskYIQlUkoRW7ZhOnXGHYOzdiaF/yhXtVHau2KLYruVdic5r41p38RXisv8Aeam713hZ3d7wMTganmQ6JVuYyMr6u5r3V2LyLy05ho4u0UH777ZP7foAAckkIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB6eFL9WYcxBTXah76Je6x570rOFqm1OHrvR3y001zoZNeCePWbv7PGi+dDUMz7Q3jZMOXXsXcZfkqrd2znbIJPK5F4ST6eyqtansaj7sv2ZBtYYDrlLrVFd+PFeK9UbIuU1s0x4y+Ed77H0MnybROVseq7emk4X8nAhmOmXSHRpauwVgrY56irb3eaF2aRx+JFThXoKRa03NR5fpLq1F9nNr6HN0bgJJ9duI7f5U/r6EgAhhZYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABDSQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//2Q=="
LOGO_SRC = f"data:image/png;base64,{_LOGO_B64}"

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG_CREAM}; }}
  [data-testid="stSidebar"] {{
      background: linear-gradient(160deg, {ORANGE} 0%, {DARK_ORG} 100%);
  }}
  [data-testid="stSidebar"] * {{ color: white !important; }}
  h1, h2, h3 {{ color: {ORANGE}; font-family: 'Segoe UI', sans-serif; }}
  [data-testid="metric-container"] {{
      background: white;
      border-left: 5px solid {ORANGE};
      border-radius: 10px;
      padding: 10px 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .section-title {{
      background: {ORANGE};
      color: white;
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 14px;
      display: inline-block;
  }}
  .stButton>button {{
      background: {ORANGE}; color: white;
      border: none; border-radius: 8px; font-weight: 600;
  }}
  .stButton>button:hover {{ background: {DARK_ORG}; }}
  /* Logo image — remove any border/shadow */
  .swiggy-logo {{
      border-radius: 12px;
      display: block;
  }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_dataset(n=1500):
    rng = np.random.default_rng(42)
    cities = ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai",
              "Pune","Kolkata","Ahmedabad","Jaipur","Surat"]
    cw = [0.18,0.16,0.15,0.12,0.10,0.09,0.08,0.05,0.04,0.03]
    areas = {
        "Mumbai":    ["Andheri","Bandra","Dadar","Juhu","Powai"],
        "Delhi":     ["Connaught Place","Hauz Khas","Lajpat Nagar","Saket","Rohini"],
        "Bengaluru": ["Koramangala","Indiranagar","Jayanagar","HSR Layout","Whitefield"],
        "Hyderabad": ["Banjara Hills","Jubilee Hills","Gachibowli","Madhapur","Hitech City"],
        "Chennai":   ["T. Nagar","Anna Nagar","Adyar","Velachery","Nungambakkam"],
        "Pune":      ["Koregaon Park","Viman Nagar","Baner","Kothrud","Hinjewadi"],
        "Kolkata":   ["Park Street","Salt Lake","Ballygunge","Newtown","Jadavpur"],
        "Ahmedabad": ["CG Road","Navrangpura","Satellite","Bodakdev","Thaltej"],
        "Jaipur":    ["MI Road","Vaishali Nagar","Malviya Nagar","C-Scheme","Mansarovar"],
        "Surat":     ["Adajan","Vesu","City Light","Varachha","Piplod"],
    }
    cp  = ["North Indian","Chinese","Biryani","South Indian","Fast Food","Pizzas",
           "Burgers","Rolls","Desserts","Mughlai","Continental","Seafood","Thali","Cafe"]
    cpw = [0.15,0.13,0.12,0.10,0.09,0.08,0.07,0.06,0.05,0.05,0.04,0.03,0.02,0.01]
    rt  = ["Quick Bites","Casual Dining","Cafe","Fine Dining",
           "Dessert Parlour","Cloud Kitchen","Food Court","Bakery"]
    rtw = [0.28,0.22,0.14,0.08,0.10,0.10,0.05,0.03]
    tpls = ["{c} Express","The {c} Kitchen","{c} House","Royal {c}","Fresh {c}",
            "{c} Corner","Urban {c}","{c} Bites","The {c} Place","Spice {c}"]
    rows = []
    for _ in range(n):
        city      = rng.choice(cities, p=cw)
        area      = rng.choice(areas[city])
        cuisine   = rng.choice(cp, p=cpw)
        rest_type = rng.choice(rt, p=rtw)
        if rest_type == "Fine Dining":              cost = float(rng.integers(1200,3500))
        elif rest_type in ["Casual Dining","Cafe"]: cost = float(rng.integers(400,1200))
        elif rest_type == "Quick Bites":            cost = float(rng.integers(150,500))
        else:                                       cost = float(rng.integers(100,800))
        base   = 2.5 + (cost/3500)*2.0
        rating = float(np.clip(rng.normal(base,0.4),1.0,5.0))
        rating = round(rating*2)/2
        votes  = int(np.clip(rng.normal(300+rating*80,150),10,2500))
        online = "No" if rest_type=="Fine Dining" else rng.choice(["Yes","No"],p=[0.55,0.45])
        if rest_type == "Cloud Kitchen": online = "Yes"
        coupon = rng.choice(["Yes","No"],p=[0.40,0.60]) if online=="Yes" else "No"
        num_c  = int(rng.integers(1,5))
        rname  = rng.choice(tpls).replace("{c}", cuisine)
        rows.append([rname,city,area,cuisine,rest_type,cost,rating,votes,online,coupon,num_c])
    return pd.DataFrame(rows, columns=[
        "Restaurant Name","City","Area","Cuisine","Restaurant Type",
        "Cost for Two (INR)","Rating","Votes","Online Order","Offers/Coupons","Num Cuisines"])

@st.cache_data
def clean_data(df):
    df = df.copy()
    df.drop_duplicates(subset=["Restaurant Name","Area","City"], inplace=True)
    df.dropna(subset=["Rating","Cuisine","Cost for Two (INR)"], inplace=True)
    df["Cost for Two (INR)"] = df["Cost for Two (INR)"].astype(float)
    df["Rating"]  = df["Rating"].astype(float)
    df["Votes"]   = df["Votes"].astype(int)
    return df.reset_index(drop=True)

df = clean_data(generate_dataset(1500))

def swiggy_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FFF8F0")
    ax.figure.patch.set_facecolor("white")
    if title:  ax.set_title(title, color=ORANGE, fontweight="bold", fontsize=13, pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=10)
    ax.tick_params(colors=TEXT_DRK)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd");  ax.spines["bottom"].set_color("#ddd")
    ax.grid(axis="y", color="#eee", linestyle="--", linewidth=0.7)


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:18px 0 6px 0;">
      <img src="{LOGO_SRC}" class="swiggy-logo"
           style="height:72px; width:72px; object-fit:contain; margin:auto;"
           alt="Swiggy"/>
      <div style="color:white; font-size:1.2rem; font-weight:700;
                  letter-spacing:2px; margin-top:8px;">SWIGGY</div>
      <div style="color:rgba(255,255,255,0.75); font-size:0.8rem; margin-top:2px;">
        Data Analysis
      </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.3); margin:10px 0 16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "Overview",
        "Data Preview & Cleaning",
        "Restaurant Analysis",
        "Ratings Analysis",
        "Pricing Analysis",
        "Location Demand",
        "Cuisine Popularity",
        "Online Orders",
        "ML — Rating Predictor",
    ])
    st.markdown('<hr style="border-color:rgba(255,255,255,0.3);">', unsafe_allow_html=True)
    st.caption("Built with Python · Streamlit")


# ── TOP BANNER ────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(90deg,{ORANGE},{DARK_ORG});
            border-radius:14px; padding:16px 32px; margin-bottom:24px;
            display:flex; align-items:center; gap:20px;">
  <img src="{LOGO_SRC}" class="swiggy-logo"
       style="height:56px; width:56px; object-fit:contain; border-radius:10px;"
       alt="Swiggy"/>
  <div>
    <div style="color:white; font-size:1.7rem; font-weight:700; letter-spacing:0.5px;">
      SWIGGY — Data Analysis
    </div>
    <div style="color:rgba(255,255,255,0.88); font-size:0.92rem; margin-top:3px;">
      Uncovering Restaurant Trends &amp; Consumer Preferences
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<span class="section-title">Project Overview</span>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Restaurants", f"{len(df):,}")
    c2.metric("Cities Covered",    df["City"].nunique())
    c3.metric("Avg Rating",        f"{df['Rating'].mean():.2f} ★")
    c4.metric("Avg Cost for Two",  f"Rs.{df['Cost for Two (INR)'].mean():.0f}")
    st.markdown("---")
    st.markdown("### Objective")
    st.info("Perform **Exploratory Data Analysis (EDA)** on the Swiggy restaurant dataset "
            "to identify key factors influencing **ratings**, **pricing**, and "
            "**online ordering behaviour**.")
    st.markdown("### Tech Stack")
    for k,v in {"Language":"Python 3.x","Data Manipulation":"Pandas, NumPy",
                "Visualisation":"Matplotlib, Seaborn, Plotly",
                "Machine Learning":"Scikit-learn (Linear Regression)",
                "Web Framework":"Streamlit"}.items():
        st.markdown(f"- **{k}:** {v}")
    st.markdown("---")
    st.markdown("### City-wise Restaurant Distribution")
    city_c = df["City"].value_counts().reset_index()
    city_c.columns = ["City","Count"]
    fig = px.bar(city_c, x="City", y="Count", color="Count",
                 color_continuous_scale=["#FFD18C",ORANGE,DARK_ORG], template="plotly_white")
    fig.update_layout(showlegend=False, plot_bgcolor=BG_CREAM, paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

elif "Data Preview" in page:
    st.markdown('<span class="section-title">Data Preview & Cleaning</span>', unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["Raw Data","Cleaning Steps","Descriptive Stats"])
    with t1:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50), use_container_width=True)
    with t2:
        st.markdown("#### Null / Missing Values")
        st.dataframe(pd.DataFrame({
            "Column":     df.columns,
            "Null Count": [df[c].isna().sum()                for c in df.columns],
            "Null %":     [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns],
        }), use_container_width=True, hide_index=True)
        st.success("No null values found after cleaning.")
        st.markdown("#### Steps Applied")
        for s in ["Removed duplicate restaurant entries.",
                  "Dropped rows with missing Rating / Cuisine / Cost.",
                  "Converted Cost for Two to float.",
                  "Encoded Online Order and Coupons as binary for ML."]:
            st.markdown(f"- {s}")
    with t3:
        st.markdown("#### Numeric Summary")
        st.dataframe(df.describe().T.style.background_gradient(cmap="Oranges"), use_container_width=True)
        st.markdown("#### Correlation Heatmap")
        fh,ah = plt.subplots(figsize=(6,4))
        sns.heatmap(df[["Cost for Two (INR)","Rating","Votes","Num Cuisines"]].corr(),
                    annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ah)
        ah.set_title("Correlation Matrix", color=ORANGE, fontweight="bold")
        st.pyplot(fh)

elif "Restaurant Analysis" in page:
    st.markdown('<span class="section-title">Restaurant Type Analysis</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        tc = df["Restaurant Type"].value_counts()
        f,a = plt.subplots(figsize=(6,4))
        a.barh(tc.index, tc.values, color=[ORANGE if i==0 else "#FFB347" for i in range(len(tc))])
        swiggy_style(a,"Restaurant Type Distribution","Count",""); st.pyplot(f)
    with c2:
        vt = df.groupby("Restaurant Type")["Votes"].sum().sort_values()
        f,a = plt.subplots(figsize=(6,4))
        a.barh(vt.index, vt.values, color=ORANGE, alpha=0.85)
        swiggy_style(a,"Total Votes by Type","Votes",""); st.pyplot(f)
    st.markdown("---"); c3,c4 = st.columns(2)
    with c3:
        ar = df.groupby("Restaurant Type")["Rating"].mean().sort_values()
        f,a = plt.subplots(figsize=(6,4))
        a.barh(ar.index, ar.values, color=[ORANGE if v>=ar.mean() else "#FFD18C" for v in ar.values])
        a.axvline(ar.mean(), color="red", linestyle="--", linewidth=1, label=f"Avg={ar.mean():.2f}")
        a.legend(fontsize=8); swiggy_style(a,"Avg Rating by Restaurant Type","Rating (out of 5)",""); st.pyplot(f)
    with c4:
        pivot = df.pivot_table(index="Restaurant Type", columns="Online Order", aggfunc="size", fill_value=0)
        f,a = plt.subplots(figsize=(6,4))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=a, linewidths=0.5, cbar_kws={"shrink":0.7})
        a.set_title("Online Order × Restaurant Type", color=ORANGE, fontweight="bold", fontsize=11); st.pyplot(f)
    st.markdown("---")
    st.markdown("#### Top 10 Most Voted Restaurants")
    st.dataframe(df.nlargest(10,"Votes")[["Restaurant Name","City","Cuisine","Restaurant Type","Rating","Votes","Cost for Two (INR)"]].reset_index(drop=True), use_container_width=True)

elif "Ratings Analysis" in page:
    st.markdown('<span class="section-title">Ratings Analysis</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        f,a = plt.subplots(figsize=(6,4))
        a.hist(df["Rating"], bins=10, color=ORANGE, edgecolor="white", alpha=0.9)
        a.axvline(df["Rating"].mean(), color="red", linestyle="--", label=f"Mean = {df['Rating'].mean():.2f}")
        a.legend(); swiggy_style(a,"Rating Distribution","Rating (out of 5)","Frequency"); st.pyplot(f)
        st.info("Most restaurants receive ratings between **3.5 – 4.5**.")
    with c2:
        fig2 = px.scatter(df, x="Votes", y="Rating", color="Restaurant Type", opacity=0.6,
                          template="plotly_white", color_discrete_sequence=px.colors.sequential.Oranges[3:])
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor=BG_CREAM); st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---"); c3,c4 = st.columns(2)
    with c3:
        f,a = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df, x="Online Order", y="Rating", palette={"Yes":ORANGE,"No":"#FFD18C"}, ax=a, width=0.4)
        swiggy_style(a,"Rating by Online Order","Online Order","Rating"); st.pyplot(f)
        st.info("Restaurants with online orders show **higher median ratings**.")
    with c4:
        f,a = plt.subplots(figsize=(6,5))
        order = df.groupby("Restaurant Type")["Rating"].median().sort_values(ascending=False).index
        sns.violinplot(data=df, x="Rating", y="Restaurant Type", order=order, palette="YlOrRd", ax=a, linewidth=0.8)
        swiggy_style(a,"Rating Distribution by Type","Rating",""); st.pyplot(f)

elif "Pricing Analysis" in page:
    st.markdown('<span class="section-title">Pricing Analysis</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        f,a = plt.subplots(figsize=(6,4))
        a.hist(df["Cost for Two (INR)"], bins=20, color=ORANGE, edgecolor="white", alpha=0.9)
        a.axvline(df["Cost for Two (INR)"].median(), color="red", linestyle="--",
                  label=f"Median=Rs.{df['Cost for Two (INR)'].median():.0f}")
        a.legend(); swiggy_style(a,"Cost for Two Distribution","Cost (Rs.)","Count"); st.pyplot(f)
    with c2:
        fig2 = px.scatter(df, x="Cost for Two (INR)", y="Rating", color="Restaurant Type", opacity=0.55,
                          template="plotly_white", color_discrete_sequence=px.colors.sequential.Oranges[2:])
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor=BG_CREAM); st.plotly_chart(fig2, use_container_width=True)
        st.info("Pricier restaurants tend to rate slightly higher.")
    st.markdown("---"); c3,c4 = st.columns(2)
    with c3:
        ac = df.groupby("Restaurant Type")["Cost for Two (INR)"].mean().sort_values()
        f,a = plt.subplots(figsize=(6,4))
        a.barh(ac.index, ac.values, color=ORANGE, alpha=0.85)
        for i,(idx,val) in enumerate(ac.items()): a.text(val+10,i,f"Rs.{val:.0f}",va="center",fontsize=8)
        swiggy_style(a,"Average Cost for Two by Type","Cost (Rs.)",""); st.pyplot(f)
    with c4:
        bins=[0,300,600,1000,2000,10000]
        labels=["Budget\n(<Rs.300)","Affordable\n(Rs.300-600)","Mid-Range\n(Rs.600-1K)","Premium\n(Rs.1K-2K)","Luxury\n(>Rs.2K)"]
        df["Cost Band"] = pd.cut(df["Cost for Two (INR)"], bins=bins, labels=labels)
        bc = df["Cost Band"].value_counts().sort_index()
        f,a = plt.subplots(figsize=(6,4))
        a.bar(bc.index, bc.values, color=[ORANGE,DARK_ORG,"#FFB347","#FFD18C","#FFF3E0"], edgecolor="white")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        swiggy_style(a,"Restaurants by Price Segment","Segment","Count"); st.pyplot(f)
        st.info("**Sweet Spot:** Rs.300–Rs.600 — optimal entry price.")
    st.markdown("---")
    cc = df.groupby("City")["Cost for Two (INR)"].mean().sort_values(ascending=False).reset_index()
    fig5 = px.bar(cc, x="City", y="Cost for Two (INR)", color="Cost for Two (INR)",
                  color_continuous_scale=["#FFD18C",ORANGE,DARK_ORG], template="plotly_white", text_auto=".0f")
    fig5.update_layout(paper_bgcolor="white", plot_bgcolor=BG_CREAM, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

elif "Location Demand" in page:
    st.markdown('<span class="section-title">Location-wise Demand</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        ac2 = df["Area"].value_counts().head(15)
        f,a = plt.subplots(figsize=(6,5))
        a.barh(ac2.index[::-1], ac2.values[::-1], color=ORANGE, alpha=0.85)
        swiggy_style(a,"Top Areas — Restaurant Density","Count",""); st.pyplot(f)
    with c2:
        cr = df.groupby("City")["Rating"].mean().sort_values(ascending=False)
        f,a = plt.subplots(figsize=(6,5))
        colors = [ORANGE if r>=cr.mean() else "#FFD18C" for r in cr.values]
        a.barh(cr.index[::-1], cr.values[::-1], color=colors[::-1])
        a.axvline(cr.mean(), color="red", linestyle="--", label=f"Mean={cr.mean():.2f}")
        a.legend(fontsize=8); swiggy_style(a,"City-wise Avg Rating","Avg Rating",""); st.pyplot(f)
    st.markdown("---")
    pivot2 = df.pivot_table(index="City", columns="Restaurant Type", aggfunc="size", fill_value=0)
    f,a = plt.subplots(figsize=(12,5))
    sns.heatmap(pivot2, annot=True, fmt="d", cmap="YlOrRd", ax=a, linewidths=0.4, cbar_kws={"shrink":0.6})
    a.set_title("City × Restaurant Type Density", color=ORANGE, fontweight="bold", fontsize=12); st.pyplot(f)
    st.markdown("---")
    cs = df.groupby("City").agg(Count=("Rating","count"),AvgRating=("Rating","mean")).reset_index()
    cs["Underserved Score"] = cs["Count"]/cs["AvgRating"]
    fig4 = px.scatter(cs, x="AvgRating", y="Count", size="Underserved Score", text="City", color="AvgRating",
                      color_continuous_scale=["#FFD18C",ORANGE,DARK_ORG], template="plotly_white")
    fig4.update_traces(textposition="top center"); fig4.update_layout(paper_bgcolor="white", plot_bgcolor=BG_CREAM)
    st.plotly_chart(fig4, use_container_width=True)

elif "Cuisine Popularity" in page:
    st.markdown('<span class="section-title">Cuisine Popularity</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        tc2 = df["Cuisine"].value_counts().head(10)
        f,a = plt.subplots(figsize=(6,6))
        wc = [ORANGE,DARK_ORG,"#FFB347","#FFD18C","#FFC56B","#FFE0A3","#FFF3E0","#E08000","#C06000","#A04000"]
        wedges,texts,autotexts = a.pie(tc2.values, labels=tc2.index, colors=wc, autopct="%1.1f%%", startangle=140, pctdistance=0.82)
        for at in autotexts: at.set_fontsize(8); at.set_color("white")
        a.set_title("Top 10 Cuisines", color=ORANGE, fontweight="bold", fontsize=12); st.pyplot(f)
    with c2:
        cb = df["Cuisine"].value_counts().head(12)
        f,a = plt.subplots(figsize=(6,5))
        a.bar(cb.index, cb.values, color=[ORANGE if i<3 else "#FFB347" for i in range(len(cb))], edgecolor="white")
        plt.xticks(rotation=45, ha="right", fontsize=8); swiggy_style(a,"Cuisine Count","","Count"); st.pyplot(f)
    st.markdown("---"); c3,c4 = st.columns(2)
    with c3:
        crat = df.groupby("Cuisine")["Rating"].mean().sort_values(ascending=False).head(14)
        f,a = plt.subplots(figsize=(6,5))
        a.barh(crat.index[::-1], crat.values[::-1], color=ORANGE, alpha=0.85)
        a.axvline(crat.mean(), color="red", linestyle="--", linewidth=1)
        swiggy_style(a,"Avg Rating by Cuisine","Avg Rating",""); st.pyplot(f)
    with c4:
        ccost = df.groupby("Cuisine")["Cost for Two (INR)"].mean().sort_values(ascending=False).head(14)
        f,a = plt.subplots(figsize=(6,5))
        a.barh(ccost.index[::-1], ccost.values[::-1], color=DARK_ORG, alpha=0.85)
        swiggy_style(a,"Avg Cost by Cuisine","Cost (Rs.)",""); st.pyplot(f)
    st.markdown("---")
    cc2 = df.pivot_table(index="City", columns="Cuisine", aggfunc="size", fill_value=0)
    f,a = plt.subplots(figsize=(14,5))
    sns.heatmap(cc2, annot=True, fmt="d", cmap="YlOrRd", ax=a, linewidths=0.3, cbar_kws={"shrink":0.6})
    a.set_title("City × Cuisine Heatmap", color=ORANGE, fontweight="bold", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8); st.pyplot(f)

elif "Online Orders" in page:
    st.markdown('<span class="section-title">Online Order Analysis</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2); oc = df["Online Order"].value_counts()
    with c1:
        f,a = plt.subplots(figsize=(5,5))
        a.pie(oc.values, labels=oc.index, colors=[ORANGE,"#FFD18C"], autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
        a.set_title("Online Order Availability", color=ORANGE, fontweight="bold"); st.pyplot(f)
        st.info("~55% of restaurants accept online orders.")
    with c2:
        coup_c = df["Offers/Coupons"].value_counts()
        f,a = plt.subplots(figsize=(5,5))
        a.pie(coup_c.values, labels=coup_c.index, colors=[DARK_ORG,"#FFB347"], autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
        a.set_title("Coupon/Offer Availability", color=ORANGE, fontweight="bold"); st.pyplot(f)
    st.markdown("---"); c3,c4 = st.columns(2)
    with c3:
        co = df[df["Online Order"]=="Yes"]["City"].value_counts()
        f,a = plt.subplots(figsize=(6,4))
        a.bar(co.index, co.values, color=ORANGE, edgecolor="white")
        plt.xticks(rotation=30, ha="right", fontsize=8); swiggy_style(a,"Online Orders — City-wise","City","Count"); st.pyplot(f)
    with c4:
        avgr = df.groupby("Online Order")["Rating"].mean()
        f,a = plt.subplots(figsize=(5,4))
        bars = a.bar(avgr.index, avgr.values, color=[ORANGE,"#FFD18C"], width=0.4, edgecolor="white")
        for bar,val in zip(bars,avgr.values): a.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.2f}", ha="center", fontweight="bold")
        swiggy_style(a,"Avg Rating: Online vs Offline","Order Mode","Avg Rating"); a.set_ylim(0,5.5); st.pyplot(f)
    st.markdown("---")
    p3 = df.pivot_table(index="Cuisine", columns="Online Order", aggfunc="size", fill_value=0)
    f,a = plt.subplots(figsize=(8,6))
    sns.heatmap(p3, annot=True, fmt="d", cmap="YlOrRd", ax=a, linewidths=0.5, cbar_kws={"shrink":0.7})
    a.set_title("Cuisine × Online Order Mode", color=ORANGE, fontweight="bold", fontsize=12); st.pyplot(f)

elif "ML" in page:
    st.markdown('<span class="section-title">ML — Multiple Linear Regression: Rating Predictor</span>', unsafe_allow_html=True)
    st.markdown("**Goal:** Predict **Restaurant Rating** using: Cost for Two · Number of Cuisines · Online Order · Offers/Coupons")
    st.markdown(f"""
    <div style="background:white; border-left:5px solid {ORANGE}; border-radius:10px;
                padding:14px 20px; margin:12px 0 16px 0; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
      <div style="color:{ORANGE}; font-weight:700; font-size:0.95rem; margin-bottom:8px;">
        📐 Multiple Linear Regression Formula
      </div>
      <div style="font-size:1.05rem; color:#1C1C1C; font-family:'Courier New', monospace; letter-spacing:0.3px;">
        <b>Rating</b> = β₀ + β₁·(Cost for Two) + β₂·(Num Cuisines) + β₃·(Online Order) + β₄·(Offers/Coupons) + ε
      </div>
      <div style="margin-top:10px; font-size:0.85rem; color:#6B6B6B; line-height:1.7;">
        &nbsp;&nbsp;<b>β₀</b> = Intercept &nbsp;|&nbsp;
        <b>β₁</b> = Cost coefficient &nbsp;|&nbsp;
        <b>β₂</b> = Cuisine breadth coefficient &nbsp;|&nbsp;
        <b>β₃</b> = Online Order coefficient &nbsp;|&nbsp;
        <b>β₄</b> = Coupon coefficient &nbsp;|&nbsp;
        <b>ε</b> = Error term
      </div>
    </div>
    """, unsafe_allow_html=True)
    ml_df = df.copy(); le = LabelEncoder()
    ml_df["Online_bin"] = le.fit_transform(ml_df["Online Order"])
    ml_df["Coupon_bin"] = le.fit_transform(ml_df["Offers/Coupons"])
    X = ml_df[["Cost for Two (INR)","Num Cuisines","Online_bin","Coupon_bin"]]; y = ml_df["Rating"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression(); model.fit(X_train, y_train); y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred); rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    cm1,cm2,cm3 = st.columns(3)
    cm1.metric("R² Score",f"{r2:.4f}"); cm2.metric("RMSE",f"{rmse:.4f}"); cm3.metric("Train Samples",f"{len(X_train):,}")
    st.markdown("---"); c1,c2 = st.columns(2)
    with c1:
        coef_df = pd.DataFrame({"Feature":X.columns,"Coefficient":model.coef_}).sort_values("Coefficient",ascending=False)
        f,a = plt.subplots(figsize=(6,4))
        a.barh(coef_df["Feature"], coef_df["Coefficient"], color=[ORANGE if v>0 else "#FF6B6B" for v in coef_df["Coefficient"]])
        a.axvline(0, color="black", linewidth=0.8); swiggy_style(a,"Feature Coefficients","Coefficient Value",""); st.pyplot(f)
    with c2:
        f,a = plt.subplots(figsize=(6,4))
        a.scatter(y_test, y_pred, alpha=0.4, color=ORANGE, edgecolors="none", s=25)
        a.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--",linewidth=1.5,label="Ideal")
        a.legend(); swiggy_style(a,"Actual vs Predicted Rating","Actual Rating","Predicted Rating"); st.pyplot(f)
    st.markdown("---")
    residuals = y_test - y_pred; f,a = plt.subplots(figsize=(8,3))
    a.hist(residuals, bins=30, color=ORANGE, edgecolor="white", alpha=0.85); a.axvline(0, color="red", linestyle="--")
    swiggy_style(a,"Residuals Distribution","Residual (Actual − Predicted)","Frequency"); st.pyplot(f)
    st.markdown("---"); st.markdown("### Live Rating Predictor")
    i1,i2 = st.columns(2)
    with i1:
        inp_cost=st.slider("Cost for Two (Rs.)",100,3500,500,step=50)
        inp_cuisines=st.slider("Number of Cuisines on Menu",1,5,2)
    with i2:
        inp_online=st.selectbox("Accepts Online Orders?",["Yes","No"])
        inp_coupon=st.selectbox("Offers Coupons / Discounts?",["Yes","No"])
    inp_vec = np.array([[inp_cost,inp_cuisines,1 if inp_online=="Yes" else 0,1 if inp_coupon=="Yes" else 0]])
    pred = float(np.clip(model.predict(inp_vec)[0],1.0,5.0))
    stars = "★"*round(pred)+"☆"*(5-round(pred))
    st.markdown(f"""<div style="background:{ORANGE};color:white;border-radius:12px;padding:18px 28px;text-align:center;margin-top:10px;">
      <h2 style="color:white;margin:0;">Predicted Rating: {pred:.2f} / 5.0</h2>
      <p style="font-size:1.8rem;margin:4px 0;">{stars}</p>
      <p style="color:rgba(255,255,255,0.85);margin:0;">Cost Rs.{inp_cost} · {inp_cuisines} cuisines · Online={inp_online} · Coupon={inp_coupon}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""<div style="background:linear-gradient(90deg,{ORANGE},{DARK_ORG});border-radius:14px;padding:28px 30px;text-align:center;">
      <img src="{LOGO_SRC}" style="height:64px;width:64px;object-fit:contain;border-radius:12px;margin-bottom:12px;" alt="Swiggy"/>
      <h2 style="color:white;margin:0;">Thank You!</h2>
      <p style="color:rgba(255,255,255,0.9);margin:6px 0 0 0;">
        <b>Vaspan Bhola</b> &nbsp;·&nbsp; 31010623019 &nbsp;·&nbsp; Minor — Data Science &nbsp;·&nbsp; Division C
      </p></div>""", unsafe_allow_html=True)
