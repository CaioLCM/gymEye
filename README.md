# 🏋️ GymEye

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

Sistema de gerenciamento de ocupação de academia por visão computacional.

Uma câmera aponta para a sala, o YOLOv8 detecta e rastreia as pessoas no
quadro, e um painel mostra em tempo real quantas estão lá dentro — sinalizando
**LIVRE** (em verde, com as vagas restantes) ou **CHEIO** (em vermelho) quando a
capacidade é atingida. A ideia é responder, sem ninguém contar na porta, à
pergunta de sempre: *vale a pena ir agora?*

O projeto é organizado em **versões de complexidade crescente**, cada uma em sua
própria pasta sob [`src/`](src/). Cada versão é autocontida e roda sozinha — dá
para comparar as abordagens lado a lado.

---

## Versões

### v1 — contagem local com painel ao vivo

A versão atual. Dois processos, acoplados apenas por um arquivo de estado:

```
worker.py  ──lê a câmera──▶  YOLOv8  ──▶  .state/ocupacao.json  ──▶  front.py
             (a cada 10s)                  (escrita atômica)        (Streamlit)
```

| Arquivo | Papel |
| --- | --- |
| [`src/v1/back.py`](src/v1/back.py) | `ContadorPessoas` mantém modelo e câmera abertos entre medições; helpers de leitura/escrita do estado |
| [`src/v1/worker.py`](src/v1/worker.py) | Produtor: mede a ocupação e publica em intervalo fixo |
| [`src/v1/front.py`](src/v1/front.py) | Painel Streamlit que se atualiza sozinho via `st.fragment` |

Separar produtor e consumidor evita que a interface trave enquanto a câmera é
lida, e mantém o modelo carregado uma única vez.

---

## Stack

| Lib | Para quê |
| --- | --- |
| **Ultralytics YOLOv8** | Detecção e tracking de pessoas (`classes=[0]`) |
| **OpenCV** | Captura de frames da câmera e dos vídeos |
| **Streamlit** | Interface do painel de ocupação |
| **uv** | Gerenciamento de dependências e execução |

---

## Como rodar

Pré-requisitos: Python 3.13+, [uv](https://docs.astral.sh/uv/) e uma webcam.

```bash
uv sync
```

Em dois terminais:

```bash
# 1. produtor — segura a câmera e publica a contagem
uv run python src/v1/worker.py

# 2. painel — só lê o estado publicado
uv run streamlit run src/v1/front.py
```

Na primeira execução o macOS pede permissão de câmera para o processo que rodou
o comando (Terminal, VS Code etc.).

### Opções do worker

```bash
uv run python src/v1/worker.py --source media/gymvideo.mp4  # usa um vídeo em vez da câmera
uv run python src/v1/worker.py --intervalo 30               # emite a cada 30s
uv run python src/v1/worker.py --janela 5                   # 5s de captura por medição
```

### Visualização com as detecções

Para ver o vídeo anotado com as caixas do YOLO:

```bash
uv run python src/v1/back.py   # 'q' encerra a janela
```

## Configuração

A capacidade máxima e o intervalo de atualização do painel ficam no topo de
[`src/v1/front.py`](src/v1/front.py):

```python
CAPACIDADE = 3   # acima disso, o painel mostra CHEIO
INTERVALO = 5    # segundos entre atualizações da tela
```
