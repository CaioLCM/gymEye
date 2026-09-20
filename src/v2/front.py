"""Painel do GymEye: cadastro, controle da academia e a visao do usuario."""

import cv2
import streamlit as st

import cadastro
import reconhecimento
from pipeline import PipelineRostos, imagem_de_bytes

CAPACIDADE = 8     # acima disso a academia conta como cheia
ATUALIZACAO = 10   # segundos entre atualizacoes das abas ao vivo

st.set_page_config(page_title="GymEye", page_icon="🏋️", layout="centered")
st.title("🏋️ GymEye")


@st.cache_resource(show_spinner="Carregando modelos...")
def carregar_pipeline():
    return PipelineRostos()


def formatar_restante(delta):
    """Tempo que falta, ou quanto ja passou do previsto."""
    minutos = int(abs(delta).total_seconds() // 60)
    texto = f"{minutos // 60}h{minutos % 60:02d}min"
    return texto if delta.total_seconds() >= 0 else f"−{texto}"


try:
    cadastro.listar_pessoas()
except Exception as erro:
    st.error(f"Sem conexao com o banco: {erro}\n\nSuba com `docker compose up -d db`.")
    st.stop()


aba_cadastro, aba_academia, aba_usuario = st.tabs(
    ["👤 Cadastro", "🏋️ Na academia", "📱 Visao do usuario"])


# --- aba 1: cadastro ------------------------------------------------------

with aba_cadastro:
    nome = st.text_input("Nome da pessoa")
    foto = st.camera_input("Rosto da pessoa")

    if foto is not None:
        with st.spinner("Procurando rosto..."):
            det = carregar_pipeline().melhor_rosto(imagem_de_bytes(foto.getvalue()))

        if det is None:
            st.warning("Nenhum rosto encontrado. Chegue mais perto da camera.")
        else:
            col_img, col_info = st.columns([1, 2])
            col_img.image(cv2.cvtColor(det.face_img, cv2.COLOR_BGR2RGB),
                          caption=f"rosto detectado ({det.face_prob:.2f})")

            # Antes de gravar, mostra com quem esse rosto ja se parece: cadastrar
            # a mesma pessoa com dois nomes estraga o reconhecimento.
            parecido = reconhecimento.identificar(det.embedding)
            if parecido and parecido["reconhecido"]:
                col_info.warning(
                    f"Esse rosto ja se parece com **{parecido['nome']}** "
                    f"({parecido['similaridade']:.2f}). Use o mesmo nome para "
                    "adicionar mais um rosto a essa pessoa."
                )

            if col_info.button("Cadastrar", type="primary", disabled=not nome.strip()):
                pessoa_id = cadastro.cadastrar(nome, det.embedding, det.face_img)
                st.success(f"**{nome}** cadastrado(a) (id {pessoa_id}).")
                st.rerun()

            if not nome.strip():
                col_info.caption("Preencha o nome para habilitar o cadastro.")

    st.divider()

    pessoas = cadastro.listar_pessoas()
    if not pessoas:
        st.info("Ninguem cadastrado ainda.")
    else:
        st.caption(f"{len(pessoas)} pessoa(s) cadastrada(s)")

        for pessoa in pessoas:
            col_img, col_txt, col_del = st.columns([1, 4, 1])

            jpeg = cadastro.rosto_de(pessoa["id"])
            if jpeg:
                col_img.image(jpeg, width=70)

            col_txt.markdown(
                f"**{pessoa['nome']}** — {pessoa['rostos']} rosto(s)  \n"
                f"<small>desde {pessoa['criado_em']:%d/%m/%Y %H:%M}</small>",
                unsafe_allow_html=True,
            )

            if col_del.button("Remover", key=f"del{pessoa['id']}"):
                cadastro.remover_pessoa(pessoa["id"])
                st.rerun()

        st.caption(
            "Dica: cadastre a mesma pessoa mais de uma vez, em angulos e luzes "
            "diferentes — cada foto vira um rosto a mais e melhora o reconhecimento."
        )


# --- aba 2: controle de quem esta na academia -----------------------------

@st.fragment(run_every=ATUALIZACAO)
def painel_academia():
    presencas = reconhecimento.presentes()
    estourados = [p for p in presencas if p["extrapolou"]]

    col_total, col_estouro = st.columns(2)
    col_total.metric("Na academia", len(presencas))
    col_estouro.metric("Tempo esgotado", len(estourados))

    if not presencas:
        st.info(
            "Ninguem reconhecido no momento. Rode o processo da camera:\n\n"
            "`uv run python src/v2/camera.py`"
        )
        return

    st.dataframe(
        [
            {
                "": "🔴" if p["extrapolou"] else "🟢",
                "Nome": p["nome"],
                "Entrada": f"{p['entrada']:%H:%M}",
                "Saida prevista": f"{p['saida_prevista']:%H:%M}",
                "Restante": formatar_restante(p["restante"]),
            }
            for p in presencas
        ],
        hide_index=True,
        width="stretch",
    )

    if estourados:
        st.error(
            f"{len(estourados)} pessoa(s) passaram do tempo previsto. "
            "A entrada delas nao e renovada enquanto a presenca nao for encerrada."
        )

    st.caption(
        f"Saida prevista = entrada + {reconhecimento.PERMANENCIA}. "
        f"Atualiza a cada {ATUALIZACAO}s."
    )

    for p in presencas:
        if st.button(f"Encerrar {p['nome']}", key=f"fim{p['pessoa_id']}",
                     type="primary" if p["extrapolou"] else "secondary"):
            reconhecimento.encerrar_presenca(p["pessoa_id"])
            st.rerun(scope="fragment")


with aba_academia:
    painel_academia()


# --- aba 3: o que o frequentador ve ---------------------------------------

@st.fragment(run_every=ATUALIZACAO)
def painel_usuario():
    """Mesma leitura da v1 — vale a pena ir agora? —, so que contando as
    pessoas reconhecidas, nao os corpos no frame."""
    total = reconhecimento.total_presentes()

    st.metric("Pessoas na academia", total)

    if total >= CAPACIDADE:
        st.markdown("<h2 style='color:red;'>CHEIO</h2>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<h2 style='color:green;'>LIVRE — {total} pessoa(s), "
            f"{CAPACIDADE - total} vaga(s) restante(s)</h2>",
            unsafe_allow_html=True,
        )

    st.progress(min(total / CAPACIDADE, 1.0))
    st.caption(f"Capacidade: {CAPACIDADE} pessoas. Atualiza a cada {ATUALIZACAO}s.")


with aba_usuario:
    painel_usuario()
