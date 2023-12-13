import os
import cv2
import torch
import numpy as np
from pyautogui import screenshot
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def emb(img):
	""" gera os embeddings de uma imagem """
	aligned = mtcnn(img)
	embeddings = resnet(aligned).detach()
	return embeddings

# Gera a lista de faces validas (que ficam verdes)
validos = []
faces = os.listdir('grupo/')
for filename in faces:
	face = cv2.imread('grupo/' + filename)
	validos.append(emb(face))

def grupo(img):
	""" verifica se uma face eh valida """
	threshold = 1.12
	emb_img = emb(img)
	difs = [ (emb_img - integrante).norm().item() for integrante in validos ]
	#print(difs)
	return any(dif <= threshold for dif in difs)

def printscreen():
	""" gera um print da tela
		excluindo os cantos superior/inferior """
	return cv2.cvtColor(np.array(screenshot(region=(0,100,1920,940))), cv2.COLOR_BGR2RGB)

margem = 30
azul = (255, 0, 0)
verde = (0, 255, 0)
vermelho = (0, 0, 255)

def detectar(img):
	""" colore os rostos da imagem
		de acordo com a validacao de grupo """
	boxes, _ = mtcnn.detect(img)
	if boxes is not None:
		for box in boxes:
			xi, yi, xf, yf = [int(x) for x in box]
			face = np.copy(img[yi-margem:yf+margem,xi-margem:xf+margem])
			try:
				cor = verde if grupo(face) else vermelho
			except Exception as e:
				cor = azul
			cv2.rectangle(img, (xi, yi), (xf, yf), cor, 1)

def scan():
	""" escaneia e mostra a tela
		com os rostos detectados e validados """
	img = printscreen()
	detectar(img)
	cv2.imshow('Tela', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

scan()