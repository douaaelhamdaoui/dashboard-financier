import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Dashboard Financier", page_icon="📈", layout="wide")

st.title("📈 Dashboard Financier")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
ticker_symbol = st.sidebar.selectbox("Choisir une action", ["AAPL", "GOOGL", "MSFT", "AMZN"])
periode = st.sidebar.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"])

# Récupérer les données
ticker = yf.Ticker(ticker_symbol)
data = ticker.history(period=periode)
info = ticker.info

# Métriques en haut
col1, col2, col3, col4 = st.columns(4)
prix_actuel = round(data["Close"].iloc[-1], 2)
prix_hier = round(data["Close"].iloc[-2], 2)
variation = round(prix_actuel - prix_hier, 2)
variation_pct = round((variation / prix_hier) * 100, 2)

col1.metric("💰 Prix actuel", f"${prix_actuel}")
col2.metric("📊 Variation", f"${variation}", f"{variation_pct}%")
col3.metric("📈 Plus haut 52s", f"${round(data['High'].max(), 2)}")
col4.metric("📉 Plus bas 52s", f"${round(data['Low'].min(), 2)}")

st.markdown("---")

# Moyennes mobiles
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# Graphique
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Prix", line=dict(color="#00b4d8", width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA 20j", line=dict(color="#f77f00", width=1.5)))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA 50j", line=dict(color="#d62828", width=1.5)))
fig.update_layout(title=f"Évolution du prix - {ticker_symbol}", template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# Tableau
st.subheader("📋 Dernières données")
st.dataframe(data[["Open","High","Low","Close","Volume"]].tail(10).round(2))

