import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="INVER100", page_icon="💹", layout="wide")

# ---------- Estilo ----------
st.markdown("""<style>
[data-testid="stApp"]{background:linear-gradient(145deg,#0b1425,#101f33);color:#edf4ff}
[data-testid="stSidebar"]{background:#0d1c30}
[data-testid="stMetric"]{background:#172b43;padding:15px;border-radius:14px;border:1px solid #29415b}
.hero{background:linear-gradient(120deg,#10243a,#14455a);border:1px solid #2dd4bf55;border-radius:22px;
padding:28px 36px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;gap:24px}
.hero h1{font-size:4.2rem;margin:0;line-height:1;color:#5eead4;letter-spacing:.04em}
.hero p{margin:.5rem 0 0;font-size:1.15rem;color:#cfe3f7}
.hero svg{max-width:340px;width:100%}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div><h1>INVER100</h1>
<p>Compara activos, mide el riesgo y descubre qué habría pasado con tus primeros 100 dólares.</p></div>
<svg viewBox="0 0 340 120"><polyline fill="none" stroke="#5eead4" stroke-width="6" stroke-linecap="round"
stroke-linejoin="round" points="8,100 60,80 110,88 160,52 215,64 270,30 332,22"/></svg></div>""",
            unsafe_allow_html=True)

# ---------- Barra lateral ----------
UNIVERSO = ["AAPL", "MSFT", "NVDA", "JPM", "GOOGL", "SPY", "GLD"]
st.sidebar.header("⚙️ Configura tu análisis")
tickers = st.sidebar.multiselect("Activos", UNIVERSO, default=["AAPL", "MSFT", "JPM"], max_selections=5)
periodo = st.sidebar.selectbox("Período histórico", ["3mo", "6mo", "1y", "2y"], index=2,
    format_func=lambda x: {"3mo": "3 meses", "6mo": "6 meses", "1y": "1 año", "2y": "2 años"}[x])
capital = st.sidebar.number_input("Capital inicial (USD)", min_value=100.0, value=10000.0, step=500.0)
ventana = st.sidebar.slider("Ventana de media móvil (días)", 5, 30, 10)
fuente = st.sidebar.radio("Fuente de datos", ["Yahoo Finance", "Simulados (práctica)"], index=0)

# ---------- Datos ----------
@st.cache_data(ttl=1800, show_spinner=False)
def descargar_precios(simbolos, periodo):
    import yfinance as yf
    datos = yf.download(list(simbolos), period=periodo, auto_adjust=True, progress=False, threads=False)
    if datos.empty:
        raise ValueError("Yahoo Finance no devolvió datos")
    if isinstance(datos.columns, pd.MultiIndex):
        cierre = datos["Close"]
    else:
        cierre = datos[["Close"]].rename(columns={"Close": simbolos[0]})
    cierre = cierre.reindex(columns=list(simbolos))
    return cierre.sort_index().ffill().dropna()

def simular_precios(simbolos):
    rng = np.random.default_rng(23)
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=180)
    return pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(.0004, .014, len(fechas))) for t in simbolos}, index=fechas)

if not tickers:
    st.info("👈 Selecciona al menos un activo en la barra lateral.")
    st.stop()

simulados = fuente == "Simulados (práctica)"
if simulados:
    precios = simular_precios(tickers)
else:
    try:
        with st.spinner("Consultando Yahoo Finance..."):
            precios = descargar_precios(tuple(tickers), periodo)
        if precios.empty or len(precios) < 30:
            raise ValueError("No hay suficientes precios comunes")
    except Exception as error:
        st.warning("Yahoo Finance no está disponible. Se muestran DATOS SIMULADOS para continuar.")
        st.caption(f"Detalle técnico: {error}")
        precios = simular_precios(tickers)
        simulados = True

if simulados:
    st.error("MODO DEMOSTRACIÓN: los resultados NO corresponden a cotizaciones reales.")
else:
    st.success("Datos históricos descargados de Yahoo Finance")

# ---------- Cálculos ----------
rendimientos = precios.pct_change().dropna()
cartera = rendimientos.mean(axis=1)                      # pesos iguales, rebalanceo diario
acumulado = (1 + cartera).cumprod()
rend_total = acumulado.iloc[-1] - 1
volatilidad = cartera.std(ddof=1) * np.sqrt(252)
valor_inv = capital * acumulado
base_100 = precios.div(precios.iloc[0]) * 100
drawdown = acumulado / acumulado.cummax() - 1

def plot(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Rendimiento acumulado", f"{rend_total:+.2%}")
c2.metric("Volatilidad anualizada", f"{volatilidad:.2%}")
c3.metric("Valor final teórico", f"USD {valor_inv.iloc[-1]:,.2f}")
st.caption("Portafolio de pesos iguales y rebalanceo diario; sin costos, impuestos ni dividendos adicionales.")

# ---------- Pestañas ----------
p1, p2, p3, p4, p5 = st.tabs(["📈 Portafolio", "📊 Estadística", "🛡️ Riesgo", "💯 INVER100", "ℹ️ Acerca de mí"])

with p1:
    plot(px.line(base_100, title="Comparación de activos · base 100"))
    media = base_100.rolling(ventana).mean().dropna()
    plot(px.line(media, title=f"Media móvil de {ventana} días"))
    izq, der = st.columns(2)
    with izq:
        plot(px.line(valor_inv.rename("USD"), title="Valor de la inversión"))
    with der:
        plot(px.bar((cartera * 100).rename("%"), title="Rendimiento diario de la cartera (%)"))
    st.info("Interpretación: la escala base 100 permite comparar trayectorias aunque los activos tengan precios distintos.")

with p2:
    resumen = pd.DataFrame({"Media diaria (%)": rendimientos.mean() * 100,
                            "Mediana diaria (%)": rendimientos.median() * 100,
                            "Desviación diaria (%)": rendimientos.std() * 100,
                            "Mínimo diario (%)": rendimientos.min() * 100,
                            "Máximo diario (%)": rendimientos.max() * 100})
    st.dataframe(resumen.round(3), width="stretch")
    a, b = st.columns(2)
    with a:
        plot(px.histogram(x=cartera * 100, nbins=30, title="Distribución de rendimientos diarios",
                          labels={"x": "Rendimiento (%)", "y": "Frecuencia"}))
    with b:
        if len(tickers) > 1:
            plot(px.imshow(rendimientos.corr(), text_auto=".2f", zmin=-1, zmax=1,
                           color_continuous_scale="RdBu_r", title="Correlaciones"))
        else:
            st.info("Elige 2 o más activos para ver la matriz de correlaciones.")
    st.info("Una correlación alta no implica causalidad. La volatilidad mide la dispersión de los rendimientos.")

with p3:
    r1, r2 = st.columns(2)
    r1.metric("Máxima caída (drawdown)", f"{drawdown.min():.2%}")
    r2.metric("Peor día de la cartera", f"{cartera.min():.2%}")
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, fill="tozeroy", line=dict(color="#f87171")))
    fig.update_layout(title="Curva de drawdown (%)")
    plot(fig)
    vol_act = (rendimientos.std() * np.sqrt(252) * 100).rename("Volatilidad anual (%)")
    plot(px.bar(vol_act, title="Volatilidad anualizada por activo (%)"))
    st.info("El drawdown muestra cuánto cae la inversión desde su punto más alto. Más volatilidad = más incertidumbre.")

with p4:
    st.subheader("💯 ¿Qué habría pasado si hubieras invertido…?")
    monto = st.number_input("Monto a invertir (USD)", min_value=10.0, value=100.0, step=10.0)
    finales = (monto * precios.iloc[-1] / precios.iloc[0]).sort_values(ascending=False)
    cols = st.columns(len(finales))
    for col, (t, v) in zip(cols, finales.items()):
        col.metric(t, f"USD {v:,.2f}", f"{v / monto - 1:+.1%}")
    plot(px.bar(finales.rename("USD"), title=f"Valor final de USD {monto:,.0f} por activo", text_auto=".2f"))
    st.markdown("#### 🎯 Reto: adivina al ganador")
    apuesta = st.radio("¿Qué activo crees que rindió más?", list(tickers), horizontal=True)
    if st.button("Revelar resultado"):
        ganador = finales.index[0]
        if apuesta == ganador:
            st.balloons()
            st.success(f"¡Acertaste! {ganador} fue el mejor y convirtió USD {monto:,.0f} en USD {finales.iloc[0]:,.2f}.")
        else:
            st.warning(f"Casi. El mejor fue {ganador} (USD {finales.iloc[0]:,.2f}). Tu elección, {apuesta}, terminó en USD {finales[apuesta]:,.2f}.")

with p5:
    st.subheader("Cómo se construyó")
    st.write("**Layout:** barra lateral, columnas y pestañas.  \n**Controles:** multiselect, selectbox, number_input, slider y radio.")
    st.write("**Datos:** yfinance y pandas.  \n**Gráficos:** Plotly.")
    st.write("**Fórmulas:** rendimiento diario, rendimiento acumulado, volatilidad anualizada (σ·√252) y drawdown.")
    st.write("**Autor:** escribe aquí tu nombre y tu interpretación.")

with st.expander("🔎 Ver datos y descargar CSV"):
    st.dataframe(precios.tail(30), width="stretch")
    st.download_button("Descargar precios CSV", precios.to_csv().encode("utf-8"), "precios_inver100.csv", "text/csv")
st.caption("Uso académico. Los resultados históricos no garantizan rentabilidades futuras.")
