"""Conexao e schema. As consultas ficam com quem tem a regra de negocio:
cadastro.py e reconhecimento.py.
"""

import os
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://gymeye:gymeye@localhost:5432/gymeye',
)

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SQL_INIT = os.path.join(RAIZ, 'sql', 'init.sql')


def conectar(url=DATABASE_URL):
    """Abre uma conexao com o tipo `vector` ja registrado."""
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


@contextmanager
def cursor(conn=None):
    """Cursor pronto para uso, com commit no fim.

    Passando `conn`, reusa a conexao e nao a fecha — e o que processos longos
    (a camera) querem, para nao reconectar a cada frame."""
    externa = conn is not None
    conn = conn or conectar()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    finally:
        if not externa:
            conn.close()


def init_schema(conn=None):
    """(Re)aplica sql/init.sql. O container aplica sozinho na primeira subida;
    isto serve para atualizar um banco que ja existia."""
    with open(SQL_INIT) as f:
        sql = f.read()

    with cursor(conn) as cur:
        cur.execute(sql)


def linhas_como_dict(cur):
    colunas = [c.name for c in cur.description]
    return [dict(zip(colunas, linha)) for linha in cur.fetchall()]
