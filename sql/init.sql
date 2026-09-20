-- Schema do GymEye. Roda automaticamente na primeira subida do container,
-- e e reaplicavel a qualquer momento via db.init_schema().
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pessoas (
    id          SERIAL PRIMARY KEY,
    nome        TEXT NOT NULL UNIQUE,
    criado_em   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Uma pessoa pode ter varios rostos cadastrados (angulos, luz, barba...),
-- o que melhora muito o reconhecimento.
CREATE TABLE IF NOT EXISTS rostos (
    id          SERIAL PRIMARY KEY,
    pessoa_id   INTEGER NOT NULL REFERENCES pessoas(id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL,
    imagem      BYTEA,
    criado_em   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cada vez que alguem e reconhecido entrando na academia.
-- A presenca fica *aberta* enquanto `saida` for NULL: passar da `saida_prevista`
-- nao fecha nada sozinho, so marca que a pessoa extrapolou o tempo.
CREATE TABLE IF NOT EXISTS presencas (
    id              SERIAL PRIMARY KEY,
    pessoa_id       INTEGER NOT NULL REFERENCES pessoas(id) ON DELETE CASCADE,
    entrada         TIMESTAMPTZ NOT NULL DEFAULT now(),
    saida_prevista  TIMESTAMPTZ NOT NULL,
    saida           TIMESTAMPTZ
);

-- Migracao para bancos criados antes da coluna `saida` existir.
ALTER TABLE presencas ADD COLUMN IF NOT EXISTS saida TIMESTAMPTZ;
ALTER TABLE presencas DROP COLUMN IF EXISTS similaridade;   -- nunca foi lida

-- No schema antigo, a presenca fechava sozinha na saida_prevista, entao a mesma
-- pessoa podia ter varias linhas em aberto. Fecha as antigas e mantem so a
-- ultima de cada pessoa, para o indice unico abaixo poder existir.
UPDATE presencas p SET saida = p.saida_prevista
WHERE p.saida IS NULL
  AND EXISTS (
      SELECT 1 FROM presencas q
      WHERE q.pessoa_id = p.pessoa_id AND q.saida IS NULL AND q.entrada > p.entrada
  );

-- Os embeddings do FaceNet ja saem com norma 1, entao distancia de cosseno
-- e a metrica natural aqui.
CREATE INDEX IF NOT EXISTS rostos_embedding_idx
    ON rostos USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS rostos_pessoa_idx ON rostos (pessoa_id);

-- No maximo uma presenca aberta por pessoa — a garantia fica no banco, nao na
-- disciplina de quem escreve.
CREATE UNIQUE INDEX IF NOT EXISTS presencas_abertas_idx
    ON presencas (pessoa_id) WHERE saida IS NULL;

DROP INDEX IF EXISTS presencas_saida_idx;   -- do schema antigo
