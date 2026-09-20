"""Visao computacional compartilhada pelo cadastro e pelo reconhecimento.

    frame --[YOLOv8]--> pessoas --[MTCNN]--> rosto alinhado --[FaceNet]--> vetor 512d

O MTCNN so procura rosto dentro do recorte da pessoa, o que reduz falso
positivo e custo por frame.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from facenet_pytorch.models.mtcnn import extract_face
from ultralytics import YOLO

CAMERA = 0
TAMANHO_ROSTO = 160   # entrada do FaceNet
MARGEM_ROSTO = 14     # px de folga em volta do rosto, no recorte alinhado
MIN_ROSTO = 40        # menor rosto que o MTCNN procura
CONF_PESSOA = 0.5     # confianca minima do YOLO


@dataclass
class Deteccao:
    """Uma pessoa detectada num frame, com o rosto e o embedding se houver."""

    person_box: tuple                   # (x1, y1, x2, y2) no frame original
    face_box: tuple | None = None       # (x1, y1, x2, y2) no frame original
    face_prob: float | None = None
    face_img: np.ndarray | None = field(default=None, repr=False)   # BGR 160x160
    embedding: np.ndarray | None = field(default=None, repr=False)  # 512d, norma 1

    @property
    def tem_rosto(self):
        return self.embedding is not None


def _dispositivo():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class PipelineRostos:
    """Carrega os tres modelos uma vez e processa frame a frame."""

    def __init__(self):
        self.device = _dispositivo()

        # O MTCNN redimensiona em piramide e cai no adaptive pooling, que o MPS
        # ainda nao implementa para tamanhos nao divisiveis — fica na CPU.
        mtcnn_device = 'cpu' if self.device == 'mps' else self.device

        self.yolo = YOLO('yolov8n.pt')
        self.mtcnn = MTCNN(
            image_size=TAMANHO_ROSTO,
            margin=MARGEM_ROSTO,
            min_face_size=MIN_ROSTO,
            keep_all=False,          # so o rosto mais proeminente por pessoa
            device=mtcnn_device,
        ).eval()
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detectar_pessoas(self, frame):
        """Etapa 1 — YOLOv8: caixas das pessoas (classe 0) no frame."""
        resultado = self.yolo(frame, classes=[0], conf=CONF_PESSOA, verbose=False)[0]

        if resultado.boxes is None or len(resultado.boxes) == 0:
            return []

        return [Deteccao(person_box=tuple(int(v) for v in box))
                for box in resultado.boxes.xyxy.cpu().numpy()]

    def recortar_rosto(self, frame, person_box):
        """Etapas 2 e 3 — MTCNN: acha o rosto dentro da pessoa e recorta alinhado.

        Retorna (tensor_alinhado, face_box_no_frame, prob) ou None."""
        x1, y1, x2, y2 = person_box
        recorte = frame[max(y1, 0):y2, max(x1, 0):x2]

        # Pessoa menor que MIN_ROSTO nao tem rosto detectavel, e a piramide de
        # escalas do MTCNN fica vazia nesse caso (torch.cat de lista vazia).
        if recorte.size == 0 or min(recorte.shape[:2]) < MIN_ROSTO:
            return None

        rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return None

        box = boxes[0]
        face = extract_face(rgb, box, image_size=TAMANHO_ROSTO, margin=MARGEM_ROSTO)

        # coordenadas do rosto de volta para o frame inteiro
        fx1, fy1, fx2, fy2 = box
        face_box = (int(x1 + fx1), int(y1 + fy1), int(x1 + fx2), int(y1 + fy2))
        return face, face_box, float(probs[0])

    def gerar_embeddings(self, faces):
        """Etapa 4 — FaceNet: tensores de rosto -> vetores 512d ja normalizados."""
        lote = torch.stack(faces).to(self.device)
        lote = (lote - 127.5) / 128.0          # fixed_image_standardization
        with torch.no_grad():
            return self.facenet(lote).cpu().numpy()

    def processar_frame(self, frame):
        """Roda a pipeline inteira num frame e devolve as deteccoes."""
        deteccoes = self.detectar_pessoas(frame)

        faces, com_rosto = [], []
        for det in deteccoes:
            achado = self.recortar_rosto(frame, det.person_box)
            if achado is None:
                continue

            face, det.face_box, det.face_prob = achado
            det.face_img = _tensor_para_bgr(face)
            faces.append(face)
            com_rosto.append(det)

        # um unico forward para todos os rostos do frame
        if faces:
            for det, emb in zip(com_rosto, self.gerar_embeddings(faces)):
                det.embedding = emb

        return deteccoes

    def melhor_rosto(self, frame):
        """O rosto mais confiavel do frame, ou None se nao houver nenhum."""
        rostos = [d for d in self.processar_frame(frame) if d.tem_rosto]
        return max(rostos, key=lambda d: d.face_prob) if rostos else None


def _tensor_para_bgr(face):
    """Tensor (3, 160, 160) em 0-255 -> imagem BGR uint8 para salvar/exibir."""
    arr = face.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def imagem_de_bytes(dados):
    """Bytes de uma foto do Streamlit -> imagem BGR do OpenCV."""
    return cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)


def para_jpeg(img):
    """Imagem BGR -> bytes JPEG, para gravar no banco."""
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ok else None
