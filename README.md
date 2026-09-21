# 🏋️ GymEye

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20+%20pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

**Controle de acesso e ocupação da academia do campus por visão computacional.**

> SECOMP 2026 — Hackathon de Visão Computacional na Logística Universitária <br>
> **Equipe 11 - Liga da Justiça:** Caio Lene, Pedro Henrique, Renan Stockler, João Pedro, Heitor Sebastião

---

## O problema

A academia do campus tem capacidade limitada e nenhum controle de quem está
dentro nem por quanto tempo. Disso saem três problemas de logística:

1. **O aluno não sabe se vale a pena ir.** Sai do laboratório, atravessa o
   campus e encontra todos os aparelhos ocupados.
2. **A gestão não sabe quem está lá dentro.** Não há registro de frequência,
   e o controle na portaria depende de alguém conferir carteirinha na mão.
3. **Controle manual de permanência.** Em horário de pico, o erro humano
   pode fazer com que alguem passe do limite e deixe pessoas de fora

Contar cabeças numa câmera resolve só o primeiro. Os outros dois exigem saber
**quem** é cada pessoa — e é aí que o reconhecimento facial entra.

**Contexto:** academia do campus universitário. **Público-alvo:** alunos e
servidores que frequentam a academia (quem consulta a ocupação) e a equipe que
administra o espaço (quem controla entradas e tempo). **Rotina:** cadastro
único no balcão; depois disso, entrar e sair sem carteirinha, catraca ou fila.

## A proposta

Uma câmera na entrada identifica quem chega, registra a entrada automaticamente
e calcula a saída prevista. A partir desse registro, o sistema serve dois
públicos na mesma aplicação:

- **A gestão** vê quem está na academia, desde quando, e quem já estourou o
  tempo previsto.
- **O aluno** vê só o que lhe interessa: **LIVRE** ou **CHEIO**, e quantas
  vagas restam.

O diferencial em relação a uma simples contagem de pessoas é que a contagem
vem de **pessoas identificadas**, não de corpos no quadro. Isso torna possível,
com o mesmo dado, controlar tempo de permanência e ter histórico de frequência
— coisas que uma contagem anônima não permite.

---

## A abordagem de visão computacional

Três modelos em cascata, cada um resolvendo um problema que o anterior não
resolve:

```
CAPTURA              PROCESSAMENTO                                  SAÍDA
────────   ──────────────────────────────────────────   ──────────────────────
                ┌──────────┐   ┌───────┐   ┌─────────┐
webcam  ──────▶ │ YOLOv8n  │──▶│ MTCNN │──▶│ FaceNet │──▶ vetor 512d
(OpenCV)        │ pessoas  │   │ rosto │   │embedding│         │
                └──────────┘   └───────┘   └─────────┘         │
                                                               ▼
                                              ┌────────────────────────────┐
                                              │ Postgres + pgvector        │
                                              │ vizinho mais próximo       │
                                              │ (cosseno, índice HNSW)     │
                                              └────────────────────────────┘
                                                               │
                                         ┌─────────────────────┴───────────┐
                                         ▼                                 ▼
                                 janela da câmera                 painel Streamlit
                              (nome, verde/vermelho)          (3 abas, atualiza 10s)
```

| Etapa | Modelo | Por que esse |
| --- | --- | --- |
| Detectar pessoas | **YOLOv8n** | Detector genérico, rápido em CPU, e a classe `person` (0) já vem treinada no COCO — não precisamos treinar nada |
| Localizar o rosto | **MTCNN** | Além de detectar, devolve o rosto **alinhado** pelos pontos faciais, que é o que o FaceNet espera na entrada |
| Descrever o rosto | **FaceNet** (InceptionResnetV1, pesos VGGFace2) | Produz um vetor de 512 dimensões em que rostos da mesma pessoa ficam próximos. Permite cadastrar alguém com **uma única foto**, sem retreinar o modelo a cada pessoa nova |
| Comparar | **pgvector** | A busca por similaridade fica no banco, junto do dado. Uma query resolve "quem é" |

**Por que não usar só o YOLO?** Ele diz que *há* uma pessoa, não *quem* é.

**Por que rodar o MTCNN dentro do recorte do YOLO**, e não no frame inteiro?
Duas razões: reduz falso positivo (padrões que lembram rosto fora de qualquer
pessoa são descartados de saída) e reduz o custo, porque a busca acontece numa
área pequena em vez da imagem toda.

**Por que cosseno?** Os embeddings do FaceNet já saem com norma 1, então a
similaridade é `1 - (a <=> b)` — o operador de distância de cosseno do pgvector.
Duas fotos da mesma pessoa ficam acima de `0.6`; pessoas diferentes, bem abaixo.

---

## Cenário de uso completo

1. **Cadastro** (uma vez por pessoa, no balcão). Na aba *Cadastro*, digita-se o
   nome e tira-se uma foto pela webcam. O sistema recorta o rosto, gera o
   embedding e grava. Antes de salvar, ele avisa se aquele rosto já se parece
   com alguém cadastrado — evita criar duas identidades para a mesma pessoa.
2. **Entrada.** Com o `camera.py` rodando na entrada, a pessoa passa na frente
   da câmera. Ela é reconhecida, a caixa fica **verde** com o nome, e a entrada
   é registrada com a saída prevista.
3. **Consulta do aluno.** Na aba *Visão do usuário*: `LIVRE — 3 pessoa(s), 5
   vaga(s) restante(s)`.
4. **Estouro de tempo.** Passada a permanência prevista, a pessoa aparece
   **vermelha** na câmera (`TEMPO ESGOTADO`) e 🔴 na aba de controle. Nenhuma
   entrada nova é criada: o tempo não se renova sozinho.
5. **Saída.** A gestão clica em *Encerrar* na aba de controle. Só então aquela
   pessoa pode registrar uma entrada nova.

---

## Arquitetura

Três processos independentes, acoplados só pelo banco:

| Processo | Papel |
| --- | --- |
| [`camera.py`](src/v2/camera.py) | Lê a câmera, reconhece e **escreve** presenças |
| [`front.py`](src/v2/front.py) | Painel Streamlit — só **lê** o banco |
| Postgres + pgvector | Guarda rostos e presenças |

Separar a captura da interface evita que o painel trave enquanto a câmera é
lida, e mantém os modelos carregados uma única vez.

### Estrutura do repositório

```
gymeye/
├── src/
│   ├── v1/                 # versão 1: contagem anônima de pessoas
│   │   ├── back.py         #   ContadorPessoas (YOLOv8)
│   │   ├── worker.py       #   produtor: mede e publica em .state/ocupacao.json
│   │   └── front.py        #   painel LIVRE/CHEIO
│   └── v2/                 # versão 2: reconhecimento facial (atual)
│       ├── pipeline.py     #   visão: pessoa → rosto → embedding
│       ├── cadastro.py     #   lógica de cadastro
│       ├── reconhecimento.py #  lógica de identificação e presença
│       ├── db.py           #   conexão e schema
│       ├── camera.py       #   processo da câmera
│       └── front.py        #   painel de 3 abas
├── sql/init.sql            # schema: pessoas, rostos, presencas
├── media/gymvideo.mp4      # vídeo de exemplo (usado na v1)
├── docker-compose.yml      # Postgres com pgvector + painel
└── Dockerfile
```

O projeto é organizado em **versões de complexidade crescente**. Cada uma roda
sozinha, o que permite comparar as duas abordagens lado a lado — a v1 mostra o
que dá para fazer sem identificar ninguém, e a v2 mostra o que a identificação
acrescenta.

### v1 — contagem anônima

```
worker.py ──lê a câmera──▶ YOLOv8 ──▶ .state/ocupacao.json ──▶ front.py
```

Conta corpos no quadro e publica um número. Sem banco, sem cadastro, sem dado
pessoal. É a linha de base: resolve o problema 1 da lista lá em cima, e só ele.

### v2 — reconhecimento facial

Duas lógicas, dois módulos: **cadastro** só cria pessoa e guarda rosto;
**reconhecimento** só identifica e registra presença. Nenhum faz o trabalho do
outro.

| Arquivo | Papel |
| --- | --- |
| [`src/v2/pipeline.py`](src/v2/pipeline.py) | Visão computacional: pessoa → rosto → embedding |
| [`src/v2/cadastro.py`](src/v2/cadastro.py) | Cadastrar, listar e remover pessoas |
| [`src/v2/reconhecimento.py`](src/v2/reconhecimento.py) | Identificar rosto, registrar entrada, listar presentes |
| [`src/v2/db.py`](src/v2/db.py) | Conexão e schema (as consultas ficam com quem tem a regra) |
| [`src/v2/front.py`](src/v2/front.py) | Painel com as três abas |
| [`src/v2/camera.py`](src/v2/camera.py) | Processo da câmera que alimenta as presenças |
| [`sql/init.sql`](sql/init.sql) | Schema: `pessoas`, `rostos`, `presencas` e o índice HNSW |

**As três abas do painel:**

- **👤 Cadastro** — nome + foto tirada na hora pela webcam, com o aviso de rosto
  parecido e a lista de cadastrados.
- **🏋️ Na academia** — visão da gestão: quem está dentro, entrada, saída
  prevista, quanto falta, 🔴 para quem estourou, e o botão de encerrar.
- **📱 Visão do usuário** — o que o aluno vê: total de pessoas, **LIVRE** em
  verde com as vagas, ou **CHEIO** em vermelho.

As duas últimas se atualizam sozinhas a cada 10s.

### Ciclo de vida de uma presença

```
reconhecido ──▶ presença aberta ──(passou da hora)──▶ 🔴 extrapolada
                      │                                     │
                      └──────── encerrar_presenca() ────────┘
                                        ▼
                                     fechada
                          (só agora pode entrar de novo)
```

Passar da `saida_prevista` **não** fecha nada sozinho. Enquanto a pessoa está
extrapolada, a câmera continua reconhecendo e pintando-a de vermelho, mas
nenhuma entrada nova é criada. Um índice único parcial
(`presencas (pessoa_id) WHERE saida IS NULL`) garante isso no banco, não na
disciplina de quem escreve o código.

---

## Como rodar

Pré-requisitos: Python 3.13+, [uv](https://docs.astral.sh/uv/), Docker e uma webcam.

```bash
uv sync
docker compose up -d db      # Postgres com pgvector, schema aplicado na 1ª subida
```

Em dois terminais:

```bash
# 1. painel — cadastro, controle e visão do usuário
uv run streamlit run src/v2/front.py

# 2. câmera — reconhece e registra as entradas ('q' encerra)
uv run python src/v2/camera.py
```

Cadastre a mesma pessoa mais de uma vez, em ângulos e luzes diferentes: cada
foto vira um rosto a mais no banco e melhora bastante o acerto.

O cadastro é só pela webcam, e o `camera.py` também precisa dela — o macOS
entrega a câmera a um processo por vez, então encerre a câmera com `q` antes de
cadastrar alguém novo.

### Ajustes

Não há argumentos de linha de comando: o que dá para mudar são constantes no
topo dos arquivos.

| Constante | Onde | Padrão |
| --- | --- | --- |
| `FONTE` | [`camera.py`](src/v2/camera.py) | `0` (webcam) — aceita o caminho de um vídeo |
| `PULAR` | [`camera.py`](src/v2/camera.py) | `3` — roda a pipeline 1 a cada N frames |
| `LIMIAR` | [`reconhecimento.py`](src/v2/reconhecimento.py) | `0.6` — similaridade mínima |
| `PERMANENCIA` | [`reconhecimento.py`](src/v2/reconhecimento.py) | tempo esperado na academia |
| `CAPACIDADE` | [`front.py`](src/v2/front.py) | `8` — acima disso, CHEIO |

### Atualizando o schema

O container aplica o [`sql/init.sql`](sql/init.sql) sozinho na primeira subida.
Num banco que já existia:

```bash
uv run python -c "import sys; sys.path.insert(0,'src/v2'); import db; db.init_schema()"
```

### Rodando o painel em container

```bash
docker compose up -d          # sobe db + front em http://localhost:8501
```

A webcam **não** é acessível de dentro do container no macOS e no Windows, e
tanto o cadastro quanto o [`camera.py`](src/v2/camera.py) dependem dela. Em
container, o painel serve para acompanhar as abas de controle e do usuário; o
cadastro e a câmera rodam no host.

### Rodando a v1

```bash
uv run python src/v1/worker.py          # produtor
uv run streamlit run src/v1/front.py    # painel
```

---

## Limitações

### A infraestrutura do local não fecha o ciclo

O sistema detecta a **entrada**, não a saída. Fechar o ciclo automaticamente
exigiria infraestrutura que a academia não tem: uma segunda câmera apontada
para quem sai, um ponto de passagem único (catraca, corredor) que obrigue todo
mundo a passar de frente, ou um sensor na porta. Do jeito que o espaço é hoje —
porta aberta, entrada e saída pelo mesmo vão, movimento em qualquer direção —
não há como distinguir alguém chegando de alguém indo embora.

A consequência prática é que **a retirada da pessoa é manual**: a presença fica
aberta até alguém clicar em *Encerrar* na aba de controle. Se a pessoa for
embora antes do tempo, a vaga continua ocupada no sistema até que a gestão a
encerre.

Foi uma escolha consciente, e não um efeito colateral: o sistema **não renova**
sozinho uma presença vencida, justamente porque não tem como saber se a pessoa
saiu. Preferimos que ela fique 🔴 na tela, visível e exigindo uma ação humana,
a fechá-la por conta própria e registrar uma saída que talvez não tenha
acontecido. Enquanto a infraestrutura não existir, o operador é a peça que
fecha o ciclo — e a interface foi desenhada para tornar isso rápido, não para
esconder que é manual.

### A identificação tem alcance limitado

**O rosto precisa estar perto.** O MTCNN só procura rostos de pelo menos 40 px.
No vídeo de exemplo ([`media/gymvideo.mp4`](media/gymvideo.mp4), 640×360), as
pessoas têm 41–89 px de largura e o rosto fica com ~15 px: a pipeline detecta as
3 pessoas e **nenhum rosto**. O sistema funciona numa câmera de **entrada**, a
cerca de 1–2 m, e não numa câmera de teto olhando a sala inteira. Sem um ponto
de passagem próximo da câmera, a taxa de reconhecimento cai muito.

**Iluminação e ângulo derrubam a similaridade.** Contraluz na porta, boné,
óculos escuros e rosto de perfil afastam o embedding do que foi cadastrado.
Cadastrar a mesma pessoa várias vezes, em condições diferentes, mitiga — não
elimina.

**O limiar de 0.6 não foi calibrado com dados reais.** Nos nossos testes, duas
variações da mesma foto deram `0.96` e um vetor aleatório deu `0.002`. A margem
é confortável nesse caso, mas entre *fotos da mesma pessoa em dias diferentes*
a separação é menor. Com poucos cadastrados, o risco maior é o falso negativo
(não reconhecer); conforme a base cresce, o falso positivo (confundir duas
pessoas) passa a pesar.

**Não há prova de vida (*anti-spoofing*).** Uma foto impressa ou na tela do
celular, colocada na frente da câmera, é reconhecida como a pessoa. Por isso o
sistema é um **apoio** ao controle de acesso, não um substituto da portaria.

---

## Trabalhos futuros

- **Câmera de saída**, fechando a presença sozinha — a peça de infraestrutura
  que hoje obriga o encerramento manual. Uma segunda câmera na saída, ou um
  ponto de passagem único, resolve.
- **Calibração do limiar** com um conjunto de teste montado no próprio campus,
  medindo falso positivo e falso negativo em vez de escolher `0.6` no olho.
- **Prova de vida** (piscada, profundidade ou textura) contra foto impressa.
- **v3: histórico e horários de pico** — a tabela `presencas` já guarda o dado
  necessário; falta a análise e a previsão de "melhor horário para ir".
- **Múltiplas câmeras** e outros espaços do campus (laboratórios, salas de
  estudo) usando a mesma base de rostos.
- **Integração com o sistema acadêmico**, reaproveitando a matrícula em vez de
  um cadastro próprio.

---

## Créditos

Todo o reconhecimento é feito com **modelos pré-treinados**; nenhum modelo foi
treinado por nós.

| Recurso | Uso no projeto | Licença |
| --- | --- | --- |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) `8.4.156` — pesos `yolov8n.pt` | Detecção de pessoas (classe COCO 0) | AGPL-3.0 |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) `2.5.3` — MTCNN + InceptionResnetV1 | Detecção/alinhamento de rosto e embedding facial | MIT |
| Pesos **VGGFace2** (via facenet-pytorch) | Modelo de embedding, pré-treinado | — |
| [PyTorch](https://pytorch.org/) `2.14.0` | Execução dos modelos | BSD-3 |
| [OpenCV](https://opencv.org/) `5.0.0` | Captura de vídeo e manipulação de imagem | Apache-2.0 |
| [pgvector](https://github.com/pgvector/pgvector) `0.5.0` + imagem `pgvector/pgvector:pg17` | Tipo `vector` e busca por similaridade | PostgreSQL License |
| [psycopg](https://www.psycopg.org/) `3.3.6` | Driver Postgres | LGPL-3.0 |
| [Streamlit](https://streamlit.io/) `1.64.0` | Interface web | Apache-2.0 |
| [uv](https://docs.astral.sh/uv/) | Dependências e execução | MIT/Apache-2.0 |

**Datasets:** nenhum dataset externo foi usado. Os rostos cadastrados são dos
próprios integrantes da equipe, coletados durante o evento. O vídeo
[`media/gymvideo.mp4`](media/gymvideo.mp4) serve apenas de exemplo para a v1.

### Uso de IA generativa

O desenvolvimento usou **Claude Code** (Anthropic) como par de programação.

- **Onde:** na escrita da pipeline de visão ([`pipeline.py`](src/v2/pipeline.py)),
  da camada de banco, do painel Streamlit e desta documentação.
- **Como:** em ciclos de especificação → implementação → teste. Cada etapa foi
  verificada rodando contra o banco e a pipeline reais (extração de embedding,
  cadastro, reconhecimento, regras de presença), e vários ajustes vieram desses
  testes — por exemplo, manter o MTCNN na CPU por falta de suporte do MPS, e
  descartar recortes menores que o tamanho mínimo de rosto, que quebravam a
  pirâmide de escalas.
- **Por quê:** o tempo do hackathon é curto e o ganho está em iterar rápido
  sobre a arquitetura. As decisões técnicas — cascata YOLO→MTCNN→FaceNet,
  embeddings no pgvector, separação cadastro/reconhecimento, presença que não se
  renova sozinha — foram tomadas e revisadas pela equipe.

### Participaçao de cada membro

- **Caio Lene:** desenvolvimento técnico e apresentação técnica no pitch
- **Pedro Henrique:** desenvolvimento dos slides e apresentação de mercado no pitch
- **Heitor Sebastião:** desenvolvimento de slides, roteiro e ideia
- **Renan Stockler:** desenvolvimento de slides, roteiro e ideia
- **João Pedro:** desenvolvimento de slides, roteiro e ideia
