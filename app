# Para rodar local: !pip install streamlit scikit-learn pandas

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Monitor de Faltas - Academia", layout="centered")

st.title("📋 Sistema de Monitoramento de Presenças - Academia Força Local")

# Base de dados fictícia
if "dados" not in st.session_state:
    alunos = ["Ana", "Beatriz", "Carlos", "Daniela", "Eduarda"]
    st.session_state.dados = pd.DataFrame({
        "aluno": alunos,
        "frequencia_no_mes": [12, 7, 9, 5, 3],
        "faltas": [2, 5, 3, 7, 9],
        "contrato_meses": [12, 2, 6, 1, 2],
        "estacao": ["inverno", "inverno", "verao", "inverno", "inverno"]
    })

# Simular registro de presença
st.subheader("🖐 Registro de Presença")
aluno = st.selectbox("Selecione o aluno (simulação de digital):", st.session_state.dados["aluno"].unique())
if st.button("Registrar presença"):
    st.session_state.dados.loc[st.session_state.dados["aluno"] == aluno, "frequencia_no_mes"] += 1
    st.success(f"Presença registrada para {aluno}")
    st.experimental_rerun()  # Atualizar a interface

# Treinar modelo simples de previsão de evasão
df = st.session_state.dados.copy()
df["estacao"] = df["estacao"].map({"inverno": 1, "verao": 0})  # binário para exemplo
df["evadiu"] = df.apply(lambda row: 1 if row["frequencia_no_mes"] < 8 and row["estacao"] == 1 else 0, axis=1)

X = df[["frequencia_no_mes", "faltas", "contrato_meses", "estacao"]]
y = df["evadiu"]

modelo = LogisticRegression()  # Instanciar modelo de regressão logística
modelo.fit(X, y)  # Treinar o modelo

# Mostrar tabela
st.subheader("📊 Dados Atuais")
st.dataframe(st.session_state.dados)

# Previsão individual
st.subheader("🔮 Parecer de Evasão")
aluno_escolhido = st.selectbox("Escolha o aluno para análise:", st.session_state.dados["aluno"].unique(), key="analisar")
linha = st.session_state.dados[st.session_state.dados["aluno"] == aluno_escolhido]
linha_proc = linha[["frequencia_no_mes", "faltas", "contrato_meses"]].copy()
linha_proc["estacao"] = 1 if linha["estacao"].values[0] == "inverno" else 0

prob = modelo.predict_proba(linha_proc)[0][1]

if prob > 0.7:
    st.error(f"🚨 Risco de evasão: ALTO ({prob:.0%})")
elif prob > 0.4:
    st.warning(f"⚠️ Risco de evasão: MÉDIO ({prob:.0%})")
else:
    st.success(f"✅ Risco de evasão: BAIXO ({prob:.0%})")
