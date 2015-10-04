#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    return cv2.imread(nombre,flagColor)

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

# Aquí cargamos imágenes con varios flags
img = leeImagen("lena.jpg",cv2.IMREAD_COLOR)
height, width = img.shape[:2]
pixels = [(y,x) for x in range(width) for y in range(height) if y % 2 != 0]
valores = [[255,255,255] for i in range(len(pixels))]

modificaPixels(img,pixels,valores)
pintaImagen("Lena",img)
