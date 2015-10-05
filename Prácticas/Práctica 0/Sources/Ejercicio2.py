#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    if flagColor == True:
        flag = cv2.IMREAD_COLOR
    else
        flag = cv2.IMREAD_GRAYSCALE
    return cv2.imread(nombre,flag)

def pintaImagen(nombre, imagen):
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Aquí cargamos imágenes con varios flags
# img = leeImagen("lena.jpg",cv2.IMREAD_GRAYSCALE)
# img = leeImagen("lena.jpg",cv2.IMREAD_COLOR)
img = leeImagen("lena.jpg",True)
pintaImagen("Lena",img)
