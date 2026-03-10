import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(page_title="Predição de Risco - Passos Mágicos", layout="wide")

# Função para carregar os modelos salvos
@st.cache_resource
def load_models():
    # Certifique-se de que esses arquivos estão na mesma pasta ou no seu GitHub
    model = joblib.load('modelo_gb_pede.joblib')
    scaler = joblib.load('scaler_pede.joblib')
    return model, scaler

# Carregando modelo e scaler
try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}. Verifique se os arquivos .joblib estão no repositório.")
    st.stop()

st.title("🎯 Sistema de Previsão de Risco do Aluno (Pergunta 9)")
st.markdown("""
Esta interface utiliza o modelo de **Gradient Boosting** para identificar padrões de risco 
baseado nos indicadores da Passos Mágicos.
""")

# --- ORGANIZAÇÃO EM 2 COLUNAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Desempenho e Engajamento")
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.5)
    ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 7.0)
    ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 8.0)

with col2:
    st.subheader("📈 Evolução e Social")
    ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.5)
    iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 9.0)
    ips = st.slider("IPS (Socioemocional)", 0.0, 10.0, 7.0)

# --- ENGENHARIA DE FEATURES ---
# Mesma lógica utilizada no treinamento do modelo (Questão 9)
gap_aprendizado = ipp - ida
suporte_total = (ips + ipp + iaa) / 3
risco_postura = 1 if (ieg < 7 and ipv < 7) else 0

# Criar DataFrame com as features na ordem exata que o modelo espera
features_list = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 
                 'GAP_APRENDIZADO', 'SUPORTE_TOTAL', 'RISCO_POSTURA']

input_df = pd.DataFrame([[
    ida, ieg, iaa, ips, ipp, ipv, 
    gap_aprendizado, suporte_total, risco_postura
]], columns=features_list)

# --- BOTÃO E PREDIÇÃO ---
st.markdown("---")
if st.button("🚀 Analisar Risco do Aluno", use_container_width=True):
    # 1. Escalar os dados
    input_scaled = scaler.transform(input_df)
    
    # 2. Predição
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 3. Exibir resultados
    if prediction == 1:
        st.error(f"### ⚠️ ALERTA DE RISCO: {probability:.1%} de probabilidade")
        st.markdown("O modelo identificou um padrão de **ALTA probabilidade** de defasagem.")
    else:
        st.success(f"### ✅ BAIXO RISCO: {probability:.1%} de probabilidade")
        st.markdown("O perfil do aluno apresenta **estabilidade** nos indicadores atuais.")

    # Mostrar métricas calculadas
    st.subheader("Indicadores Calculados")
    m1, m2, m3 = st.columns(3)
    m1.metric("Gap Aprendizado", f"{gap_aprendizado:.2f}")
    m2.metric("Suporte Total", f"{suporte_total:.2f}")
    m3.metric("Risco Postura", "Sim" if risco_postura == 1 else "Não")

st.sidebar.info("Ajuste os sliders para simular o comportamento do aluno e clique no botão de análise.")