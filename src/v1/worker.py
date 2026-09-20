"""Produtor: le a camera continuamente e publica a ocupacao a cada 10s."""

import argparse
import time

from back import CAMERA, ContadorPessoas, escrever_ocupacao


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=CAMERA,
                        help='indice da camera (padrao 0) ou caminho de um video')
    parser.add_argument('--intervalo', type=float, default=10,
                        help='segundos entre cada emissao (padrao 10)')
    parser.add_argument('--janela', type=float, default=3,
                        help='segundos de captura usados em cada medicao (padrao 3)')
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    with ContadorPessoas(source) as contador:
        print(f'worker iniciado (fonte={source!r}, intervalo={args.intervalo}s)')
        while True:
            inicio = time.time()
            pessoas = contador.contar(args.janela)
            escrever_ocupacao(pessoas)
            print(f'{time.strftime("%H:%M:%S")}  pessoas={pessoas}')

            espera = args.intervalo - (time.time() - inicio)
            if espera > 0:
                time.sleep(espera)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nworker encerrado')
