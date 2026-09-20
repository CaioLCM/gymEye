"""Logica de reconhecimento: dado um rosto, dizer de quem e e registrar a
presenca na academia.

Uma presenca fica aberta ate alguem encerra-la. Passar da `saida_prevista` nao
fecha nada: a pessoa so passa a contar como *extrapolada*, e nenhuma entrada
nova e criada enquanto isso.

Nada aqui cria pessoa nova — isso e cadastro.py.
"""

from datetime import timedelta

import numpy as np

import db

# Acima disto consideramos a mesma pessoa. Calibre com os seus proprios dados:
# rostos da mesma pessoa costumam ficar em 0.7+, pessoas diferentes abaixo de 0.5.
LIMIAR = 0.6

# Quanto tempo se espera que a pessoa fique na academia.
PERMANENCIA = timedelta(minutes=1)


def identificar(embedding, conn=None):
    """Vizinho mais proximo do embedding entre os rostos cadastrados.

    Retorna {'pessoa_id', 'nome', 'similaridade', 'reconhecido'}, ou None se
    nao ha ninguem cadastrado."""
    vetor = np.asarray(embedding, dtype=np.float32)

    with db.cursor(conn) as cur:
        cur.execute(
            """
            SELECT p.id, p.nome, 1 - (r.embedding <=> %s) AS similaridade
            FROM rostos r
            JOIN pessoas p ON p.id = r.pessoa_id
            ORDER BY r.embedding <=> %s
            LIMIT 1
            """,
            (vetor, vetor),
        )
        linha = cur.fetchone()

    if linha is None:
        return None

    pessoa_id, nome, similaridade = linha
    return {
        'pessoa_id': pessoa_id,
        'nome': nome,
        'similaridade': float(similaridade),
        'reconhecido': float(similaridade) >= LIMIAR,
    }


def marcar_presenca(pessoa_id, conn=None):
    """Registra a entrada da pessoa, se ela ainda nao estiver na academia.

    Retorna a presenca com 'nova': False significa que ela ja estava dentro —
    seja dentro do prazo, seja extrapolada. Uma presenca vencida *nao* e
    renovada; ela precisa ser encerrada antes."""
    with db.cursor(conn) as cur:
        cur.execute(
            """
            SELECT id, entrada, saida_prevista, now() > saida_prevista AS extrapolou
            FROM presencas
            WHERE pessoa_id = %s AND saida IS NULL
            """,
            (pessoa_id,),
        )
        abertas = db.linhas_como_dict(cur)
        if abertas:
            return {**abertas[0], 'nova': False}

        cur.execute(
            """
            INSERT INTO presencas (pessoa_id, saida_prevista)
            VALUES (%s, now() + %s)
            RETURNING id, entrada, saida_prevista
            """,
            (pessoa_id, PERMANENCIA),
        )
        id_, entrada, saida_prevista = cur.fetchone()

    return {'id': id_, 'entrada': entrada, 'saida_prevista': saida_prevista,
            'extrapolou': False, 'nova': True}


def reconhecer(embedding, conn=None):
    """Identifica o rosto e, se for alguem conhecido, garante a presenca dele.

    Retorna o resultado de identificar() com 'presenca' (None se nao
    reconhecido)."""
    resultado = identificar(embedding, conn=conn)
    if resultado is None:
        return None

    resultado['presenca'] = (
        marcar_presenca(resultado['pessoa_id'], conn=conn)
        if resultado['reconhecido'] else None
    )
    return resultado


def presentes():
    """Quem esta na academia agora, extrapolados inclusive."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT p.id AS pessoa_id,
                   p.nome,
                   pr.entrada,
                   pr.saida_prevista,
                   pr.saida_prevista - now() AS restante,
                   now() > pr.saida_prevista AS extrapolou
            FROM presencas pr
            JOIN pessoas p ON p.id = pr.pessoa_id
            WHERE pr.saida IS NULL
            ORDER BY pr.entrada
            """
        )
        return db.linhas_como_dict(cur)


def total_presentes():
    """Quantas pessoas estao na academia agora."""
    with db.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM presencas WHERE saida IS NULL')
        return cur.fetchone()[0]


def encerrar_presenca(pessoa_id):
    """Marca a saida da pessoa agora. So depois disso ela pode entrar de novo."""
    with db.cursor() as cur:
        cur.execute(
            'UPDATE presencas SET saida = now() '
            'WHERE pessoa_id = %s AND saida IS NULL',
            (pessoa_id,),
        )
