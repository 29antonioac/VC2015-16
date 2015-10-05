#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def leeImagen(nombre, flagColor):
    if flagColor == True:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    return cv2.imread(nombre,flag)

# Aquí cargamos imágenes con varios flags
img = leeImagen("lena.jpg",True)

cv2.imshow("Lena",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
