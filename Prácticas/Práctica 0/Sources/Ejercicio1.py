#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    return cv2.imread(nombre,flagColor)

# Aquí cargamos imágenes con varios flags
# img = leeImagen("lena.jpg",cv2.IMREAD_GRAYSCALE)
# img = leeImagen("lena.jpg",cv2.IMREAD_COLOR)
img = leeImagen("lena.jpg",cv2.IMREAD_UNCHANGED)

cv2.imshow("Lena",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
