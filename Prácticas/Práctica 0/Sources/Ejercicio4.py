#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    if flagColor == True:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    return cv2.imread(nombre,flag)

def pintaImagen(nombre, imagen):
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Supongamos que pixels es una listas de tuplas (y,x) y valores una lista de valores
def modificaPixels(imagen,pixels,valores):
    if len(pixels) != len(valores):
        return "No hay el mismo número de píxels que de valores"
    for i in range(len(pixels)):
        imagen[pixels[i][0],pixels[i][1]] = valores[i]

# Cargar imagen
img = leeImagen("lena.jpg",True)
height, width = img.shape[:2]
pixels = [(y,x) for x in range(width) for y in range(height) if y % 2 != 0]
valores = [[255,255,255] for i in range(len(pixels))]

modificaPixels(img,pixels,valores)
pintaImagen("Lena",img)
