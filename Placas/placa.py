import cv2
import pytesseract
import numpy as np

def detectar_y_leer_placa(fragmento):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(fragmento, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para reducir el ruido mientras se preservan los bordes, aplicando un vecindario de 11px, un sigma de color 17px y un sigma de espacio de 17px
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    
    # Aplicar detección de bordes usando Canny (variable, umbral inferior, umbral superior) dentro de ese rango se detectará los bordes
    bordes = cv2.Canny(gris, 30, 200)
    
    ## Aplicar transformaciones morfológicas para cerrar pequeñas brechas en los bordes detectados
    # Con la primera función de cv2 creamos un elemento estruturante (matriz morfológica), con la segunda indicamos que el elemento debe ser rectangular (como una placa)...
    # y se establece un tamaño de 5x5 px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # La primera función realiza una operación morfológica, la segunda función de cv2 cierra pequeños agujeros dentro de los objetos.
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
    
    # Se coloca el "_" porque el segundo valor que nos devuelve cv2.findContours no lo necesitamos. 
    # cv2.findContours: Encontrar contornos, cv2.RETR_TREE: Recupera todos los contornos y los rescontruye en una jerarquia de contornos anidados
    # cv2.CHAIN_APPROX_SIMPLE : Aproxima los contornos comprimiendolos de manera horizontal, verticales y diagonales  
    contornos, _ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos más grandes. (contornos es una lista)
    # cv2.contourArea : calcula el área del contorno, y con el código completo indica que se debe ordenar de acuerdo al área de cada contorno
    # reverse=True : Ordenar en orden descendente.... y solo los 10 más grandes
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10] 

    # Inicializar la variable de la placa
    placa = None

    # Iterar a través de los contornos para encontrar una forma rectangular (posible placa)
    for contorno in contornos:
        # cv2.approxPolyDP : aproximación de polígonos...  cv2.arcLength : calcula la longitud de contorno
        # 0.018 * cv2.arcLength(contorno, True) : Al multiplicar la longitud establece la distancia máxima permitida entre el contorno y el polígono y colocamos True para indicar que el contorno es cerrado
        aproximacion = cv2.approxPolyDP(contorno, 0.019 * cv2.arcLength(contorno, True), True)
        if len(aproximacion) == 4: # Se valida que tenga 4 lados 
            placa = aproximacion
            break

    if placa is None:
        return fragmento, "No se encontro ninguna placa."
    else:
        # Dibujar el contorno de la placa en la imagen original con color azul eléctrico
        cv2.drawContours(fragmento, [placa], -1, (0, 0, 255), 3)

        # Crear una máscara y recortar la placa de la imagen
        mascara = np.zeros(bordes.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [placa], 0, 255, -1)
        imagen_mascarada = cv2.bitwise_and(fragmento, fragmento, mask=mascara)

        # Recortar la región de la placa
        (x, y) = np.where(mascara == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        recorte = imagen_mascarada[topx:bottomx+1, topy:bottomy+1]

        # Convertir la imagen recortada a escala de grises y aplicar umbralización
        recorte_gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        _, recorte_binario = cv2.threshold(recorte_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Usar Tesseract para leer el texto de la placa
        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(recorte_binario, config=config)
        texto = texto.strip().replace('\n', '').replace('\x0c', '').replace(' ', '').upper()
        
        # Guardar el texto de la placa en un archivo
        with open("placas.txt", "a") as archivo:
            archivo.write(texto + "\n")
        
        return fragmento, texto

def principal():
    # Capturar video desde la cámara
    captura = cv2.VideoCapture(1)

    while True:
        # Leer un fragmento del video
        ret, fragmento = captura.read()
        if not ret:
            break

        # Detectar y leer la placa en el fragmento
        fragmento, texto = detectar_y_leer_placa(fragmento)

        # Mostrar el texto detectado en color azul eléctrico
        cv2.putText(fragmento, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el fragmento con la placa detectada
        cv2.imshow('Placa detectada', fragmento)

        # Salir del loop al presionar la tecla 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):
            break

    # Liberar el objeto de captura y cerrar todas las ventanas
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    principal()
