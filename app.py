import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(page_title="Predição de Risco - Passos Mágicos", layout="wide")

# Função para carregar os modelos salvos
@st.cache_resource
def load_models():
    # Carrega o modelo (RandomForest) e o Scaler salvos no notebook
    model = joblib.load('modelo_gb_pede.joblib')
    scaler = joblib.load('scaler_pede.joblib')
    return model, scaler

# Carregando modelo e scaler
try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}. Verifique se os arquivos .joblib estão no mesmo diretório do app.py.")
    st.stop()

st.title("🎯 Sistema de Previsão de Risco do Aluno (Questão 9)")
st.markdown("""
Esta interface utiliza o modelo de **Random Forest** treinado para identificar o risco de defasagem 
com base nos indicadores de performance e comportamento.
""")

# --- ORGANIZAÇÃO EM 2 COLUNAS PARA OS SLIDERS ---
st.subheader("📋 Insira os Indicadores do Aluno")
col1, col2 = st.columns(2)

with col1:
    ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 7.0)
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.0)
    ian = st.slider("IAN (Adequação de Nível)", 0.0, 10.0, 8.0)
    inde = st.slider("INDE (Índice Geral)", 0.0, 10.0, 7.5)

with col2:
    ips = st.slider("IPS (Socioemocional)", 0.0, 10.0, 7.5)
    ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 7.5)
    ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.5)

# --- PREPARAÇÃO DOS DADOS ---
# A ordem das colunas deve ser EXATAMENTE a mesma usada no X_train do notebook:
# ['IDA', 'IEG', 'IPS', 'IPP', 'IPV', 'IAN', 'INDE']
features_list = ['IDA', 'IEG', 'IPS', 'IPP', 'IPV', 'IAN', 'INDE']

input_df = pd.DataFrame([[
    ida, ieg, ips, ipp, ipv, ian, inde
]], columns=features_list)

# --- BOTÃO E PREDIÇÃO ---
st.markdown("---")
if st.button("🚀 Analisar Risco de Defasagem", use_container_width=True):
    # 1. Escalar os dados (usando o scaler treinado no notebook)
    input_scaled = scaler.transform(input_df)
    
    # 2. Realizar a Predição
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 3. Exibir resultados de forma visual
    st.subheader("Resultado da Análise")
    
    if prediction == 1:
        st.error(f"### ⚠️ ALERTA: Risco de Defasagem Detectado")
        st.write(f"O modelo estima uma probabilidade de **{probability:.1%}** para este perfil entrar em risco.")
        st.info("Recomendação: Intervenção pedagógica imediata e acompanhamento psicossocial.")
    else:
        st.success(f"### ✅ Perfil Estável: Baixo Risco")
        st.write(f"A probabilidade de risco calculada é de apenas **{probability:.1%}**.")
        st.balloons()

# Sidebar com informações adicionais
st.sidebar.header("Sobre o Modelo")
st.sidebar.write("""
Este modelo foi treinado com dados históricos da Passos Mágicos (2020-2022). 
Ele analisa o impacto combinado de notas, engajamento e fatores psicopedagógicos.
""")
st.sidebar.divider()
st.sidebar.info("Certifique-se de que o arquivo `requirements.txt` contenha `scikit-learn`, `joblib`, `pandas` e `streamlit`.")