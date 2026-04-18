import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="⚽ Football Analytics", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">⚽ Football Analytics Dashboard</p>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Filtros")

# Competencias
comp_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
df_comp = pd.DataFrame(requests.get(comp_url).json())

comp_selected = st.sidebar.selectbox("Competencia", df_comp["competition_name"].unique())
comp_filtered = df_comp[df_comp["competition_name"] == comp_selected]

season_selected = st.sidebar.selectbox("Temporada", comp_filtered["season_name"])
row = comp_filtered[comp_filtered["season_name"] == season_selected].iloc[0]

competition_id = row["competition_id"]
season_id = row["season_id"]

# Partidos
matches_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
df_matches = pd.json_normalize(requests.get(matches_url).json())

df_matches["match_name"] = (
    df_matches["home_team.home_team_name"].fillna("Local") +
    " vs " +
    df_matches["away_team.away_team_name"].fillna("Visitante")
)

match_selected = st.sidebar.selectbox("Partido", df_matches["match_name"])
match_id = df_matches[df_matches["match_name"] == match_selected]["match_id"].values[0]

# =========================
# EVENTOS
# =========================
events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
df = pd.json_normalize(requests.get(events_url).json())

shots = df[df["type.name"] == "Shot"].copy()
shots = shots[[
    "team.name", "player.name", "shot.outcome.name",
    "shot.statsbomb_xg", "location"
]]

shots["x"] = shots["location"].apply(lambda loc: loc[0])
shots["y"] = shots["location"].apply(lambda loc: loc[1])

# =========================
# FILTRO JUGADOR
# =========================
jugadores = shots["player.name"].dropna().unique()

jugador_seleccionado = st.sidebar.selectbox(
    "Jugador",
    ["Todos"] + sorted(jugadores)
)

shots_filtrados = shots if jugador_seleccionado == "Todos" else shots[shots["player.name"] == jugador_seleccionado]

# =========================
# MÉTRICAS GENERALES
# =========================
xg_por_equipo = shots_filtrados.groupby("team.name")["shot.statsbomb_xg"].sum().reset_index()

goles = shots_filtrados[shots_filtrados["shot.outcome.name"] == "Goal"] \
    .groupby("team.name").size().reset_index(name="Goles")

resumen = pd.merge(xg_por_equipo, goles, on="team.name", how="left").fillna(0)

# =========================
# TARJETAS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("⚽ Goles", int(resumen["Goles"].sum()))
col2.metric("📊 xG", round(resumen["shot.statsbomb_xg"].sum(), 2))
col3.metric("🎯 Tiros", len(shots_filtrados))

# =========================
# TABLAS
# =========================
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📊 Resumen")
    st.dataframe(resumen)

with col2:
    st.subheader("🔥 Top jugadores")
    top_jugadores = shots_filtrados.groupby("player.name")["shot.statsbomb_xg"].sum().sort_values(ascending=False).head(5)
    st.dataframe(top_jugadores.reset_index())

# =========================
# MAPA DE TIROS
# =========================
st.subheader("🗺️ Mapa de tiros por equipo")

equipos = shots_filtrados["team.name"].unique()
cols = st.columns(len(equipos))

for i, equipo in enumerate(equipos):
    equipo_shots = shots_filtrados[shots_filtrados["team.name"] == equipo]

    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlim(0,120)
    ax.set_ylim(0,80)

    # cancha
    ax.plot([0,0],[0,80]); ax.plot([0,120],[80,80])
    ax.plot([120,120],[80,0]); ax.plot([120,0],[0,0])
    ax.plot([60,60],[0,80])

    for _, shot in equipo_shots.iterrows():
        x, y = shot["x"], shot["y"]

        if x < 60:
            x = 120 - x
            y = 80 - y

        color = "green" if shot["shot.outcome.name"] == "Goal" else "red"
        ax.scatter(x, y, s=shot["shot.statsbomb_xg"]*1000, c=color, alpha=0.6)

    ax.set_title(equipo)
    ax.axis("off")
    cols[i].pyplot(fig)

# =========================
# HEATMAP POR JUGADOR
# =========================
st.subheader("🔥 Heatmap jugador")

if jugador_seleccionado != "Todos":
    data = shots[shots["player.name"] == jugador_seleccionado]

    if len(data) > 2:
        x,y = [],[]
        for _, shot in data.iterrows():
            sx, sy = shot["x"], shot["y"]
            if sx < 60:
                sx = 120 - sx
                sy = 80 - sy
            x.append(sx); y.append(sy)

        xi, yi = np.mgrid[0:120:100j, 0:80:100j]
        zi = gaussian_kde([x,y])(np.vstack([xi.flatten(), yi.flatten()]))

        fig, ax = plt.subplots(figsize=(6,5))
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', alpha=0.7)
        ax.set_title(jugador_seleccionado)
        ax.axis("off")
        st.pyplot(fig)

# =========================
# COMPARACIÓN
# =========================
st.sidebar.subheader("🆚 Comparar jugadores")

jugador_1 = st.sidebar.selectbox("Jugador 1", sorted(jugadores), key="j1")
jugador_2 = st.sidebar.selectbox("Jugador 2", sorted(jugadores), key="j2")

def calcular_metricas(data):
    goles = len(data[data["shot.outcome.name"] == "Goal"])
    xg = data["shot.statsbomb_xg"].sum()
    tiros = len(data)

    tiros_puerta = len(data[
        (data["shot.outcome.name"] == "Goal") |
        (data["shot.outcome.name"] == "Saved")
    ])

    xg_por_tiro = xg / tiros if tiros else 0
    conversion = goles / tiros if tiros else 0
    eficiencia = goles / xg if xg else 0
    volumen = tiros + xg

    return [goles,xg,tiros,tiros_puerta,xg_por_tiro,conversion,eficiencia,volumen]

labels = ["Goles","xG","Tiros","Tiros puerta","xG/Tiro","Conv","Efic","Volumen"]

m1 = calcular_metricas(shots[shots["player.name"] == jugador_1])
m2 = calcular_metricas(shots[shots["player.name"] == jugador_2])

max_vals = [max(a,b) if max(a,b)!=0 else 1 for a,b in zip(m1,m2)]
m1 = [a/b for a,b in zip(m1,max_vals)]
m2 = [a/b for a,b in zip(m2,max_vals)]

angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
angles += angles[:1]
m1 += m1[:1]; m2 += m2[:1]

fig, ax = plt.subplots(subplot_kw=dict(polar=True))
ax.plot(angles,m1,label=jugador_1)
ax.fill(angles,m1,alpha=0.25)
ax.plot(angles,m2,label=jugador_2)
ax.fill(angles,m2,alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.legend()

st.pyplot(fig)