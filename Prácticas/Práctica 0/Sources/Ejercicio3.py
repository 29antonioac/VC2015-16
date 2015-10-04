#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    return cv2.imread(nombre,flagColor)

def pintaSecuenciaImagenes(nombres,imagenes):
    if len(nombres) != len(imagenes):
        return "No hay el mismo número de nombres que de imágenes"

    for i in range(len(imagenes)):
        cv2.imshow(nombres[i],imagenes[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Aquí cargamos imágenes con varios flags
img1 = leeImagen("lena.jpg",cv2.IMREAD_GRAYSCALE)
img2 = leeImagen("lena.jpg",cv2.IMREAD_COLOR)
img3 = leeImagen("lena.jpg",cv2.IMREAD_UNCHANGED)
nombres = ["img1","img2","img3"]
imagenes = [img1,img2,img3]
pintaSecuenciaImagenes(nombres,imagenes)
