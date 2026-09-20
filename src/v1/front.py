import time

import streamlit as st

from back import ler_ocupacao

CAPACIDADE = 3
INTERVALO = 5

st.set_page_config(page_title="GymEye", page_icon="🏋️")
st.title("🏋️ GymEye")


@st.fragment(run_every=INTERVALO)
def painel():
    estado = ler_ocupacao()

    if estado is None:
        st.info(
            "Nenhuma leitura ainda. Rode o worker em outro terminal:\n\n"
            "`uv run python src/v1/worker.py`"
        )
        return

    pessoas = estado["pessoas"]
    st.metric("Pessoas na academia", pessoas)

    if pessoas >= CAPACIDADE:
        st.markdown("<h2 style='color:red;'>CHEIO</h2>", unsafe_allow_html=True)
    else:
        vagas = CAPACIDADE - pessoas
        st.markdown(
            f"<h2 style='color:green;'>LIVRE — {pessoas} pessoas, "
            f"{vagas} vaga(s) restante(s)</h2>",
            unsafe_allow_html=True,
        )

    idade = time.time() - estado["ts"]
    if idade > INTERVALO * 3:
        st.warning(f"Leitura de {int(idade)}s atras — o worker pode estar parado.")
    else:
        st.caption(f"Atualizado ha {int(idade)}s")


painel()
