"""Processo da camera: reconhece quem aparece e registra a entrada no banco.

    camera -> pipeline -> reconhecimento.reconhecer -> presencas

O front so le a tabela `presencas`. Encerra com 'q'.
"""

import cv2

import db
import reconhecimento
from pipeline import CAMERA, PipelineRostos

FONTE = CAMERA      # ou o caminho de um video: 'media/gymvideo.mp4'
PULAR = 3           # roda a pipeline 1 a cada N frames
JANELA = 'GymEye — reconhecimento'

VERDE = (0, 200, 0)         # reconhecido, dentro do tempo
VERMELHO = (0, 0, 220)      # extrapolou o tempo previsto
CINZA = (150, 150, 150)     # desconhecido ou sem rosto


def cor_e_texto(resultado):
    """Como pintar a pessoa: verde dentro do prazo, vermelho se extrapolou."""
    if resultado is None:
        return CINZA, 'sem rosto'

    if not resultado['reconhecido']:
        return CINZA, f"? {resultado['nome']} ({resultado['similaridade']:.2f})"

    presenca = resultado['presenca']
    if presenca['extrapolou']:
        return VERMELHO, f"{resultado['nome']} — TEMPO ESGOTADO"

    return VERDE, f"{resultado['nome']} ({resultado['similaridade']:.2f})"


def desenhar(frame, det, resultado):
    """Caixa da pessoa + etiqueta com o nome."""
    x1, y1, x2, y2 = det.person_box
    cor, texto = cor_e_texto(resultado)

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

    if det.face_box:
        fx1, fy1, fx2, fy2 = det.face_box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), cor, 1)

    (largura, altura), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - altura - 8), (x1 + largura + 6, y1), cor, -1)
    cv2.putText(frame, texto, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    # Uma conexao para todo o processo: reconectar a cada frame seria o gargalo.
    conn = db.conectar()
    pipe = PipelineRostos()
    cap = cv2.VideoCapture(FONTE)
    if not cap.isOpened():
        raise RuntimeError(f'Nao foi possivel abrir a fonte de video: {FONTE!r}')

    print(f"device={pipe.device}  fonte={FONTE!r}  'q' encerra.")

    marcados, i = [], 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if i % PULAR == 0:
                marcados = [
                    (det, reconhecimento.reconhecer(det.embedding, conn=conn)
                     if det.tem_rosto else None)
                    for det in pipe.processar_frame(frame)
                ]

                for _det, resultado in marcados:
                    if resultado and resultado['presenca'] and resultado['presenca']['nova']:
                        presenca = resultado['presenca']
                        print(f"entrada: {resultado['nome']} as "
                              f"{presenca['entrada']:%H:%M:%S} — saida prevista "
                              f"{presenca['saida_prevista']:%H:%M:%S}")

            for det, resultado in marcados:
                desenhar(frame, det, resultado)

            cv2.imshow(JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nencerrado')
