# %% [markdown]
# # Desarrollo

# %% [markdown]
# #### Importamos los módulos

# %%
import numpy as np
import random

# %% [markdown]
# #### Función para crear tableros

# %%
def crear_tablero (dimension_tablero):
    tablero = np.full((dimension_tablero,dimension_tablero), " ")
    return tablero

# %% [markdown]
# #### Creamos la funcion para colocar los barcos

# %%
#Con esta función podremos crear un barco nuevo
def crear_barco (tablero,longitud_barco):
    #Definimos las coordenadas del disparo de manera aleatoria. Deberán ser enteros que vayan de 0 hasta como máximo 
    #la dimensión del tablero. El método random.randint considera el segundo límite como inclusivo. Por eso restamos uno.
    indice_limite = tablero.shape[0] - 1

    #Creamos una flag que nos permitirá salir del bucle while cuando se den las condiciones de los if's incluidos más abajo
    valido = False

    #Creamos un contador que nos permitirá salir del bucle cuando queden pocas posiciones y el programa no encuentre
    #espacio libre para un barco. Esto evitará que el bucle sea infinito
    contador = 0
    
    while valido == False:
        contador += 1 #Sumamos uno al bucle cada vez que se recorre
        orientacion = random.choice(["N","S","E","O"]) #Almacenamos la orientacion (Norte, Sur...) en una variable. Se elige aleatoriamente
        
        #Definimos un punto de partida aleatorio a partir del que se crea el barco en su coordenada fila e iniciamos el punto final
        punto_origen_fila = random.randint(0,indice_limite)
        punto_final_fila = punto_origen_fila

        #Definimos un punto de partida aleatorio a partir del que se crea el barco en su coordenada columna e iniciamos el punto final
        punto_origen_columna = random.randint(0,indice_limite)
        punto_final_columna = punto_origen_columna

        #Según la orientación, se re-definen los puntos origen y final
        #Es importante que la coordenada origen sea menor que la coordenada final siempre
        #Esto se debe a como se va a acceder al tablero, donde usa la forma matriz[fila_origen:fila_final,columna_origen:columna:final]
        #Si fila origen fuese menor que fila final, el output sería una matriz nula de 0 dimensiones
        if orientacion == "N":
            punto_origen_fila = punto_origen_fila - (longitud_barco) 
            punto_origen_columna = punto_origen_columna
            punto_final_columna = punto_final_columna + 1 #Sumamos uno porque si fuese de [a:a], el retorno sería una matriz nula 
        if orientacion == "S":
            punto_final_fila = punto_origen_fila + (longitud_barco)
            punto_origen_columna = punto_origen_columna
            punto_final_columna = punto_final_columna + 1
        if orientacion == "O":
            punto_origen_columna = punto_origen_columna - (longitud_barco)
            punto_final_columna = punto_final_columna
            punto_final_fila = punto_final_fila + 1
        if orientacion == "E":
            punto_final_columna = punto_origen_columna + (longitud_barco)
            punto_origen_fila = punto_origen_fila
            punto_final_fila = punto_final_fila + 1
        
        #El bucle itera 1000 veces. Si no encuentra forma de colocar el barco en 1000 intentos, saldrá del bucle.
        if contador == 1000:
            print("Tras 1000 intentos, no ha sido posible colocar un nuevo barco. Comprueba si es posible añadir uno nuevo manualmente: \n")
            valido = True

        """Mientras el contador sea menor que 1000, se creará un barco y se saldrá del bucle cuando:
            1. Todos los puntos origen y fin estén dentro de las dimensiones de la matriz
            2. No haya ningun barco en algunos de los puntos donde se vaya a crear
        """

        if (0 <= punto_final_fila <= indice_limite+1) and (0 <= punto_final_columna <= indice_limite+1) and (0 <= punto_origen_fila <= indice_limite+1) and (0 <= punto_origen_columna <= indice_limite+1) and "O" not in tablero[punto_origen_fila:punto_final_fila,punto_origen_columna:punto_final_columna]:
            valido = True
            tablero[punto_origen_fila:punto_final_fila,punto_origen_columna:punto_final_columna] = "O"
        
    return
    

# %% [markdown]
# #### Creamos la funcion que nos va a crear los barcos aleatoriamente

# %%
def crear_tablero_jugador():
    tablero = crear_tablero(10)
    # Cada lista de la lista representa [Numero de barcos, Longitud del barco]
    lista_barcos = [[4,1],[3,2],[2,3],[1,4]]

    for barco in lista_barcos:
        for i in range (0,barco[0],1):
            crear_barco(tablero=tablero, longitud_barco=barco[1])    
    #Ahora añadimos una fila y una columna para que sea mas facil leerlos cuando se muestren

    tablero = np.vstack([np.arange(1,11),tablero])
    tablero = np.hstack([np.arange(0,11).reshape(11,1),tablero])
    return tablero

# %% [markdown]
# #### Creamos una clase para crear jugadores

# %%
class Jugador():
    def __init__(self, id_jugador):
        self.id_jugador = id_jugador
        self.tablero = crear_tablero_jugador()
        #Las vidas serán el número total de posiciones de barcos que se tiene inicialmente
        self.vidas = list(self.tablero.ravel()).count("O")
    
    def reducir_vidas(self):
        self.vidas = self.vidas - 1
        print(f"Al jugador {self.id_jugador} le quedan {self.vidas} vidas.")
    
    def mostrar_tablero (self):
        print(self.tablero)
    
    #Devuelve los disparos que le han hecho al jugador pero oculta los barcos
    def mostrar_historial_disparos (self):
        copia = self.tablero.copy()
        shape = copia.shape
        copia = list(copia.ravel())
        for index, value in enumerate (copia):
            if value == "O":
                copia[index] = " "
        copia = np.array(copia).reshape(shape)
        print(copia)

# %% [markdown]
# #### Funcion para crear disparos aleatorios para la máquina

# %%
#Esta funcion cambiará el tablero en función de donde caiga el disparo
#Siempre devuelve el tablero completo. Por eso solo se puede usar por la maquina ya que
#Si la usa el jugador mostrará los barcos de la máquina
def disparo_aleatorio(jugador_objetivo):
    
    #Definimos las coordenadas del disparo de manera aleatoria. Deberán ser enteros que vayan de 0 hasta como máximo 
    #la dimensión del tablero. El método random.randint considera el segundo límite como inclusivo. Por eso restamos uno.
    indice_limite = jugador_objetivo.tablero.shape[0] - 1

    coordenada_fila_disparo = random.randint(1,indice_limite)
    coordenada_columna_disparo = random.randint(1,indice_limite)

    #Creamos un bucle, de tal manera que el disparo aleatorio no elija una celda ya elegida anteriormente

    while jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "-" or jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "X":
        coordenada_fila_disparo = random.randint(1,indice_limite)
        coordenada_columna_disparo = random.randint(1,indice_limite)   
    #Si la coordenada coincide con una celda vacia, caerá en el agua
    if jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == " ":
        print("El disparo ha caido en el agua \n")
        jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] = "-"
        jugador_objetivo.mostrar_tablero()
        return False
    #Si coincide con una celda llena, caerá en un barco
    elif jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "O":
        print("El disparo ha tocado un barco \n")
        jugador_objetivo.reducir_vidas()
        jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] = "X"
        jugador_objetivo.mostrar_tablero()
        print("La IA dispara de nuevo. \n")
        return True

# %% [markdown]
# #### Funcion para crear los disparos manuales del jugador

# %%
#Las coordenadas irán de 1 a 10 en lugar de de 0 a 9, debido a que hemos metido una fila y columna de guia
#En el disparo manual, tras disparar se mostrará el historial de disparos y no el tablero del jugador
#Esto es porque el disparo manual siempre será usado por el jugador. Si muestra el tablero del otro jugador o de la maquina
#El juego pierde la gracia

def disparo_manual(jugador_objetivo):
    
    #Creamos esta variable para que entre en bucle. Despues la pasamos a True para salir. 
    #Solo se vuelve False y repite el bucle si el usuario dispara en una celda donde ya disparó
    #Ver ultimo if de esta funcion
    disparo_correcto = False
    while disparo_correcto == False:
        disparo_correcto = True

        check_flag = False
        while check_flag == False:

            #Primero comprobamos que no se introducen "Floats"
            coordenada_fila_disparo = input("Coordenada fila (1-10): ")
            while "," in coordenada_fila_disparo or "." in coordenada_fila_disparo:
                input("Coordenada incorrecta. Introduzca de nuevo. \nCoordenada fila: ")
            
            try:
                coordenada_fila_disparo = int(coordenada_fila_disparo)
                if 1 <= coordenada_fila_disparo <= 10:
                    check_flag = True
                else:
                    print("Coordenada fuera de los límites. Introduzca de nuevo.\n")
            except:
                print("Coordenada incorrecta. Introduzca de nuevo")
            
        check_flag = False
        while check_flag == False:

            #Primero comprobamos que no se introducen "Floats"
            coordenada_columna_disparo = input("Coordenada columna (1-10): ")
            while "," in coordenada_columna_disparo or "." in coordenada_columna_disparo:
                input("Coordenada incorrecta. Introduzca de nuevo. \nCoordenada columna: ")
            
            try:
                coordenada_columna_disparo = int(coordenada_columna_disparo)
                if 1 <= int(coordenada_columna_disparo) <= 10:
                    check_flag = True
                else:
                    print("Coordenada fuera de los límites. Introduzca de nuevo.\n")           
            except:
                print("Coordenada incorrecta. Introduzca de nuevo")
                
        #Si la coordenada coincide con una celda vacia, caerá en el agua
        if jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == " ":
            print("El disparo ha caido en el agua \n")
            jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] = "-"
            jugador_objetivo.mostrar_historial_disparos()

            return False
        
        #Si coincide con una celda llena, caerá en un barco
        elif jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "O":
            print("El disparo ha tocado un barco \n")
            jugador_objetivo.reducir_vidas()
            jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] = "X"
            jugador_objetivo.mostrar_historial_disparos()
            print("Dispara de nuevo. \n")

            return True
        
        elif jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "-" or jugador_objetivo.tablero[coordenada_fila_disparo,coordenada_columna_disparo] == "X":
            print("Ya se ha disparado en esa celda. Prueba de nuevo \n")
            jugador_objetivo.mostrar_historial_disparos()
            disparo_correcto = False

        

# %% [markdown]
# #### Creamos la clase main, donde se ejecuta el juego

# %% [markdown]
# Nos va a pedir como input el nombre de usuario

# %%
def main ():
    nombre_usuario = input("Introduce el nombre del jugador: ")

    jugador = Jugador(id_jugador = nombre_usuario)
    maquina = Jugador(id_jugador = "IA")
    declaracion_victoria = False

    print("Este es tu tablero: \n")
    jugador.mostrar_tablero()
    print()
    while declaracion_victoria == False:
                

        valid_options = ["1","2","3","4","5"]
        print("\n---Menu---")
        print("Introduce 1 para hacer un disparo")
        print("Introduce 2 para ver tu Tablero")
        print("Introduce 3 para ver tu historial de disparos (Tablero rival)")
        print("Introduce 4 para ver el numero de vidas del rival")
        print("Introduce 5 para ver tus vidas\n")

        menu = input("--> ").strip()
        
        while menu not in valid_options:
            print("Opcion introducida no valida.\n")
            print("\n---Menu---")
            print("Introduce 1 para hacer un disparo")
            print("Introduce 2 para ver tu Tablero")
            print("Introduce 3 para ver tu historial de disparos")
            print("Introduce 4 para ver el numero de vidas del rival")
            print("Introduce 5 para ver tus vidas\n")
            menu = input("--> ").strip()

        if menu == "1":
            while disparo_manual(jugador_objetivo=maquina) == True and maquina.vidas > 1:
                pass

            print("\n----Turno de la IA----\n")
            while disparo_aleatorio(jugador_objetivo=jugador) == True and jugador.vidas >1:
                pass
            
            print()

        elif menu == "2":
            print("\nEste es tu tablero: \n")
            jugador.mostrar_tablero()
            print()
        
        elif menu == "3":
            print("\nEste es el historial de tus disparos: \n")
            maquina.mostrar_historial_disparos()
            print()
        
        elif menu == "4":
            if maquina.vidas > 1:
                print(f"\nA tu rival le quedan {maquina.vidas} vidas.\n")
            if maquina.vidas == 1:
                print(f"\nA tu rival le queda una (1) vida.\n")
        
        elif menu =="5":
            if jugador.vidas > 1:
                print(f"\nTe quedan {maquina.vidas} vidas.\n")
            if jugador.vidas == 1:
                print(f"\nSólo te queda una (1) vida.\n")
    
    if jugador.vidas == 0:
        print("\nNo te quedan más barcos. Has perdido")
        declaracion_victoria = True
    if maquina.vidas == 0:
        print("\n¡A tu rival no le quedan más barcos! ¡Victoria!")
        declaracion_victoria = True

# %% [markdown]
# 

# %%
main()

# %%



