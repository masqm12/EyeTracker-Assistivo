import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
from pynput.mouse import Button, Controller

mouse = Controller()
mp_face_mesh = mp.solutions.face_mesh
# Offset de calibração
calibration_offset_yaw = 0
calibration_offset_pitch = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Número de frames para suavização do "raio"
filter_length = 8

if not cap.isOpened():
    print("não achou a câmera")

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Índices de landmarks importantes para o cubo e direção da cabeça
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}
# Buffers para armazenar os últimos dados do raio (suavização)
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)
fila_vertical_cima = deque(maxlen=20)
fila_vertical_baixo = deque(maxlen=20)
def projetar(pt3d):
        return int(pt3d[0]), int(pt3d[1])

# Índices dos pontos do contorno da face no MediaPipe FaceMesh
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]  
OLHO_DIREITO_PONTOS = [442, 450]

OLHO_ESQUERDO_PONTOS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]


def aplicar_filtro_vermelho_inplace(frame, intensidade=0.5):
    """
    Aplica filtro vermelho no frame original, alterando ele diretamente.
    """
    overlay = np.zeros_like(frame)
    overlay[:] = (0, 0, 255)  # vermelho puro

    # altera frame in-place
    frame[:] = cv2.addWeighted(frame, 1-intensidade, overlay, intensidade, 0)

# Converte landmark do MediaPipe para array numpy com coordenadas 3D
def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])



def desenhar_cubo(landmarks, largura, altura):
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(landmarks.landmark[idx], largura, altura)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # desenha landmarks chave     
   
        # Extrai os pontos para conveniência
        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]
 
        # Calcula eixos orientados baseados na geometria da cabeça
        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)#calcula o tamanho do vetor
        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)
        forward_axis = np.cross(right_axis, up_axis)#calcula o produto vetorial 
        forward_axis /= np.linalg.norm(forward_axis)
        # Garante que o vetor forward saia da face
        forward_axis = -forward_axis
        # Calcula centro da cabeça
        center = (left + right + top + bottom + front) / 5 #calcula o ponto médio de 5 valores
        # Metade dos tamanhos do cubo (width, height, depth)
        half_width = np.linalg.norm(right - left) / 2
        half_height = np.linalg.norm(top - bottom) / 2
        half_depth = 80  # pode ajustar ou calcular se tiver landmark de trás
           # Função para gerar cantos do cubo alinhados com a cabeça
        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * half_depth * forward_axis)

        # Lista dos 8 cantos do cubo
        #os sinais certos para gerar todos os 8 vértices do cubo.
        cube_corners = [
            corner(-1, 1, -1),   # topo-esquerda-frente
            corner(1, 1, -1),    # topo-direita-frente
            corner(1, -1, -1),   # baixo-direita-frente
            corner(-1, -1, -1),  # baixo-esquerda-frente
            corner(-1, 1, 1),    # topo-esquerda-trás
            corner(1, 1, 1),     # topo-direita-trás
            corner(1, -1, 1),    # baixo-direita-trás
            corner(-1, -1, 1)    # baixo-esquerda-trás
        ]

        # Projeta pontos 3D para coordenadas 2D da tela
        def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])

        # Desenha linhas do cubo (wireframe)
        #a função project converte os pontos 3d para 2d
        cube_corners_2d = [project(pt) for pt in cube_corners]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # frente
            (4, 5), (5, 6), (6, 7), (7, 4),  # trás
            (0, 4), (1, 5), (2, 6), (3, 7)   # laterais
        ]
        for i, j in edges:
            cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], (255, 125, 35), 2)

        # Atualiza buffers para suavização do raio
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        # Calcula média da origem e direção do raio
        avg_origin = np.mean(ray_origins, axis=0)   
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)  # normaliza

        # Direção de referência (camera olhando reto)
        reference_forward = np.array([0, 0, -1])
        
        # Ângulo horizontal (yaw) em relação à câmera
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # esquerda é negativa

        # Ângulo vertical (pitch) em relação à câmera
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # cima é positiva

        # Converte radianos para graus
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)
        yaw = yaw_deg
        pitch = pitch_deg
        # Ajuste para 0-360
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg
        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg



        # Limites de quanto a tela cobre em graus
        yawDegrees = 20
        pitchDegrees = 10

        # Aplica offsets de calibração
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch

        # Mapeia ângulos para posição da tela
        screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * largura)
        screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * altura)
        
        # Ajuste para 0-360
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg
        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

    
        # Limites de quanto a tela cobre em graus
        yawDegrees = 20
        pitchDegrees = 10

        # Aplica offsets de calibração
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch
        MONITOR_WIDTH = largura
        MONITOR_HEIGHT = altura
        print(yaw_deg, pitch_deg)
        # Mapeia ângulos para posição da tela
        screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
        screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)


        #print(f"Screen position: x={screen_x}, y={screen_y}")
        #print(yaw, pitch)
      
        # Desenha o raio médio suavizado
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        print(ray_end)
        return yaw, pitch, ray_end
       #cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
    

while True:
    retorno, frame = cap.read()
    if not retorno:
        print("o frame não foi bem sucedido")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    saida = face_mesh.process(rgb_frame)
    altura, largura, _ = frame.shape
    if saida.multi_face_landmarks:
        rosto = saida.multi_face_landmarks[0]
        yawpitch = desenhar_cubo(rosto, largura, altura)
        marcos = rosto.landmark
        pupila_direita_x = int(marcos[468].x * largura)
        pupila_direita_y = int(marcos[468].y * altura)
        pupila_esquerda_x = int(marcos[473].x * largura)
        pupila_esquerda_y = int(marcos[473].y * altura)
        pupila_esquerda_np = np.array([int(marcos[473].x * largura), int(marcos[473].y * altura), int(marcos[473].z * largura)])
        cv2.putText(frame, "pe", (pupila_esquerda_x, pupila_esquerda_y), 1,1,(255,0,0),1,1 )
        #cv2.line(frame, (pupila_esquerda_x, pupila_esquerda_y),
        #         (pupila_direita_x, pupila_direita_y), (0, 255, 0), 1)
        canto_lateral_direito_x = int(marcos[33].x * largura)
        canto_lateral_direito_y = int(marcos[33].y * altura)
        canto_lateral_esquerdo_x = int(marcos[359].x * largura)
        canto_lateral_esquerdo_y = int(marcos[359].y * altura)
        cv2.circle(frame, (canto_lateral_direito_x,
                   canto_lateral_direito_y), 2, (0, 0, 255), 10)
        cv2.circle(frame, (canto_lateral_esquerdo_x,
                   canto_lateral_esquerdo_y), 2, (0, 0, 255), 10)
        linha_direita = math.dist((pupila_direita_x, pupila_direita_y),
                                  (canto_lateral_direito_x, canto_lateral_direito_y))
        print("distancia direita", linha_direita)
        linha_esquerda = math.dist((pupila_esquerda_x, pupila_esquerda_y),
                                   (canto_lateral_esquerdo_x, canto_lateral_esquerdo_y))
        print("distancia esquerda", linha_esquerda)
        cv2.circle(frame, (pupila_direita_x, pupila_direita_y), 2, (0, 0, 255), 2)
        cv2.circle(frame, (pupila_esquerda_x, pupila_esquerda_y), 2, (0, 0, 255), 2)
        print("pupila direita:", pupila_direita_x, ":", pupila_direita_y)
        distancia_cima = marcos[473].y - marcos[475].y 
        distancia_baixo =  marcos[253].y - marcos[473].y 
        fila_vertical_cima.append(distancia_cima)
        fila_vertical_baixo.append(distancia_baixo)
        if len(fila_vertical_baixo) >= 20:
            media_vertical_cima = np.mean(fila_vertical_cima)
            media_vertical_baixo = np.mean(fila_vertical_baixo)

            if(distancia_cima) < media_vertical_cima:
                print("olhando para cima")
                
            print("media cima", media_vertical_cima)
            print("media baixo", media_vertical_baixo)

        print("distancia_cima", distancia_cima)
        print("distancia_baixo", distancia_baixo) 
        

        r = range(27, 30)
 

    


        if (linha_direita > linha_esquerda):
            print("olhando para esquerda")
            #mouse.move(-1, 0)
        else:
            print("olhando para direita")
            #mouse.move(1, 0)
        if yawpitch[0] < -10:
            print("Rosto virado para ESQUERDA")

        elif yawpitch[0] > 10:
            print("Rosto virado para DIREITA")

        else:
            print("Rosto CENTRADO")
        if yawpitch[1] < -10:
            print("Rosto virado para CIMA")

        elif yawpitch[1] > 10:

            print("Rosto virado para BAIXO")
        else:
            print("Rosto CENTRADO")

        bolha_horizontal = int(yawpitch[0])
        bolha_vertical = int(yawpitch[1])
        print("bolha_vertical:", bolha_vertical)
        fim_do_raio = projetar(yawpitch[2])
        print("pitch", yawpitch[1])
        if (abs(bolha_horizontal) < 160 or abs(bolha_vertical) < 170):
            aplicar_filtro_vermelho_inplace(frame, intensidade=0.4)
        else:
            print("na bolha")

        pts = np.array([[marcos[i].x * largura, marcos[i].y * altura,
                       marcos[i].z * largura] for i in OLHO_DIREITO_PONTOS])
      

        media_olho_esquerdo = np.mean(pts, axis=0, dtype=int)
        media_olho_esquerdo_np = np.array([media_olho_esquerdo[0], int(media_olho_esquerdo[1]), int(media_olho_esquerdo[2])])
        cv2.putText(frame, "media", (media_olho_esquerdo_np[0], media_olho_esquerdo_np[1]), 1,1,(255,0,0),1,1 )
        cv2.circle(frame, (int(media_olho_esquerdo[0]), int(
            media_olho_esquerdo[1])), 2, (0, 0, 0), 4)
        for k in OLHO_DIREITO_PONTOS:
            cv2.circle(frame, (int(marcos[k].x *largura), int(marcos[k].y * altura)), 1, (0,0,0),2)
        try:

            vetor = pupila_esquerda_np - media_olho_esquerdo_np
           
            fator = 5
            vetor_alongado = vetor * fator
            ponto_final = media_olho_esquerdo_np + vetor_alongado
     
        

            cv2.arrowedLine(frame, (media_olho_esquerdo_np[0], media_olho_esquerdo_np[1]), (int(ponto_final[0]) , int(ponto_final[1])), (255,0,0), 2,2 )
        
        except Exception as e:
            print(e) 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cap.release()
cv2.destroyAllWindows()
