import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(page_title="Predição de Risco - Passos Mágicos", layout="wide")

# Função para carregar os modelos salvos
@st.cache_resource
def load_models():
    model = joblib.load('modelo_gb_pede.joblib')
    scaler = joblib.load('scaler_pede.joblib')
    return model, scaler

model, scaler = load_models()

st.title("🎯 Sistema de Previsão de Risco do Aluno (Pergunta 9)")
st.markdown("""
Esta interface utiliza o modelo de **Gradient Boosting** treinado para identificar padrões de risco 
baseado nos indicadores da Passos Mágicos.
""")

# Layout com colunas para os parâmetros
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Indicadores Principais")
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.5)
    ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 7.0)
    ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 8.0)
    ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.5)

with col2:
    st.subheader("📈 Autoavaliação e Social")
    iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 9.0)
    ips = st.sidebar.slider("IPS (Socioemocional)", 0.0, 10.0, 7.0)

# Engenharia de Features (Mesma lógica do notebook)
gap_aprendizado = ipp - ida
suporte_total = (ips + ipp + iaa) / 3
risco_postura = 1 if (ieg < 7 and ipv < 7) else 0

# Criar DataFrame com as features na ordem correta
features_list = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 
                 'GAP_APRENDIZADO', 'SUPORTE_TOTAL', 'RISCO_POSTURA']

input_df = pd.DataFrame([[
    ida, ieg, iaa, ips, ipp, ipv, 
    gap_aprendizado, suporte_total, risco_postura
]], columns=features_list)

# Botão de ação
if st.button("🚀 Analisar Risco"):
    # 1. Escalar os dados
    input_scaled = scaler.transform(input_df)
    
    # 2. Predição
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 3. Exibir resultados
    st.divider()
    if prediction == 1:
        st.error(f"### ⚠️ ALERTA DE RISCO: {probability:.1%} de probabilidade")
        st.markdown("O modelo identificou um padrão de risco de defasagem para este perfil.")
    else:
        st.success(f"### ✅ BAIXO RISCO: {probability:.1%} de probabilidade")
        st.markdown("O perfil do aluno apresenta estabilidade nos indicadores analisados.")

    # Mostrar métricas calculadas internamente
    st.subheader("Análise Interna das Features Calculadas")
    m1, m2, m3 = st.columns(3)
    m1.metric("Gap Aprendizado", f"{gap_aprendizado:.2f}")
    m2.metric("Suporte Total", f"{suporte_total:.2f}")
    m3.metric("Risco Postura", "Sim" if risco_postura == 1 else "Não")

st.sidebar.info("As features 'Gap Aprendizado', 'Suporte Total' e 'Risco Postura' são calculadas automaticamente com base nos sliders.")