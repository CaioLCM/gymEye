from ultralytics import YOLO
import cv2

import json
import os
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIDEO_PATH = os.path.join(ROOT, 'media', 'gymvideo.mp4')
STATE_PATH = os.path.join(ROOT, '.state', 'ocupacao.json')
CAMERA = 0


class ContadorPessoas:
    """Mantem o modelo e a camera abertos entre as medicoes."""

    def __init__(self, source=CAMERA):
        self.source = source
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Nao foi possivel abrir a fonte de video: {source!r}')

    def contar(self, segundos=5):
        """Le a fonte por alguns segundos e retorna o maior numero de
        pessoas visto em um frame."""
        maximo = 0
        fim = time.time() + segundos
        while time.time() < fim:
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model.track(frame, persist=True, classes=[0], verbose=False)
            boxes = results[0].boxes
            maximo = max(maximo, 0 if boxes is None else len(boxes))

        return maximo

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def contar_pessoas(source=CAMERA, segundos=5):
    """Medicao avulsa: abre a fonte, conta e fecha."""
    with ContadorPessoas(source) as contador:
        return contador.contar(segundos)


def escrever_ocupacao(pessoas, path=STATE_PATH):
    """Grava o estado de forma atomica, para o front nunca ler JSON pela metade."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {'pessoas': pessoas, 'ts': time.time()}

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise

    return payload


def ler_ocupacao(path=STATE_PATH):
    """Retorna o ultimo estado publicado pelo worker, ou None se nao houver."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def analyse_gym(source=CAMERA):
    # Load a pre-trained YOLO model (e.g., YOLOv8 nano)

    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, classes=[0])

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow('YOLO Person Recognition', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyse_gym()
