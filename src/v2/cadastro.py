"""Logica de cadastro: guardar o rosto de uma pessoa no banco.

Nada aqui decide *quem e* alguem — isso e reconhecimento.py.
"""

import numpy as np

import db
from pipeline import para_jpeg


def cadastrar(nome, embedding, face_img):
    """Cria a pessoa (ou reusa a existente) e guarda mais um rosto dela."""
    nome = (nome or '').strip()
    if not nome:
        raise ValueError('O nome nao pode ser vazio.')

    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pessoas (nome) VALUES (%s)
            ON CONFLICT (nome) DO UPDATE SET nome = EXCLUDED.nome
            RETURNING id
            """,
            (nome,),
        )
        pessoa_id = cur.fetchone()[0]

        cur.execute(
            'INSERT INTO rostos (pessoa_id, embedding, imagem) VALUES (%s, %s, %s)',
            (pessoa_id, np.asarray(embedding, dtype=np.float32), para_jpeg(face_img)),
        )

    return pessoa_id


def listar_pessoas():
    """Todas as pessoas cadastradas, com quantos rostos cada uma tem."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.nome, COUNT(r.id) AS rostos, p.criado_em
            FROM pessoas p
            LEFT JOIN rostos r ON r.pessoa_id = p.id
            GROUP BY p.id
            ORDER BY p.nome
            """
        )
        return db.linhas_como_dict(cur)


def rosto_de(pessoa_id):
    """Primeira imagem de rosto da pessoa, como bytes JPEG (ou None)."""
    with db.cursor() as cur:
        cur.execute(
            'SELECT imagem FROM rostos WHERE pessoa_id = %s AND imagem IS NOT NULL '
            'ORDER BY id LIMIT 1',
            (pessoa_id,),
        )
        linha = cur.fetchone()
        return bytes(linha[0]) if linha and linha[0] else None


def remover_pessoa(pessoa_id):
    """Apaga a pessoa e, em cascata, os rostos e as presencas dela."""
    with db.cursor() as cur:
        cur.execute('DELETE FROM pessoas WHERE id = %s', (pessoa_id,))
