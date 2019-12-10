import threading    # Libreria usada para manejar multiples hilos por proceso
import time         # Libreria usada para dormir el hilo principal
import random
import socket       # Libreria usada para comunicacion entre procesos
import struct       # Libreria para manejar bytes como datos desempacados
import numpy as np  # Libreria para manejar las matrices
import re           # Libreria para manjera expresiones regulares
import ast          # Libreria para usa Abstract Syntax Trees
from pandas import read_csv, DataFrame # Leer csvs
from federatedPCA import SAPCA,merge # Algoritmos implementados segun el paper
from sklearn.preprocessing import scale

def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


def send(data, port=5007, addr='224.1.1.1'):
    """send(data[, port[, addr]]) - multicasts a UDP datagram."""
    # Create the socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Make the socket multicast-aware, and set TTL.
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    # Send the data
    s.sendto(data, (addr, port))

def recv(port=5007, addr="224.1.1.1", buf_size=1024):
        """recv([port[, addr[,buf_size]]]) - waits for a datagram and returns the data."""

        # Create the socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,socket.IPPROTO_UDP)

        # Set some options to make it multicast-friendly
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass # Some systems don't support SO_REUSEPORT
        s.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)
        s.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)
        s.settimeout(random.randint(4,8))

        # Bind to the port
        s.bind(('', port))

        mreq = struct.pack("4sl", socket.inet_aton(addr), socket.INADDR_ANY)

        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        # Receive the data, then unregister multicast receive membership, then close the port
        data,send_addr = s.recvfrom(buf_size)
        s.setsockopt(socket.SOL_IP, socket.IP_DROP_MEMBERSHIP, socket.inet_aton(addr) + socket.inet_aton('0.0.0.0'))
        s.close()
        return data

def Participante(currentU,currentS,currentR,q):
    # Bandera para indicar si el proceso ya fue participante en algun momento.
    # en caso de serlo la ejecucion se detiene.
    yaFuiParticipante = False
    print("Soy participante")
    while soyParticipante:
        try:
            invitacion=recv()
            puertoDeMiLider=int(invitacion.split()[1])
            # Llego una invitacion de otro lider.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Se inicia la comuniciacion para transmitir las estimaciones actuales.
            s.connect((TCP_IP, puertoDeMiLider))
            cadAEnviar=np.array_str(currentU)+"/"+np.array_str(currentS)+"/"+str(currentR)
            # Se envia el mensaje codificado
            s.send(cadAEnviar)
            s.close()
            # Se finaliza la comunicacion y se establece la bandera como verdadera.
            yaFuiParticipante = True
            break
        except socket.timeout:
            continue
        except Exception as e:
            print("Excepcion de participacion", e)
    q.append(yaFuiParticipante)
    return

soyParticipante = False 
TCP_IP = '127.0.0.1' # Direccion IP local para comunicacion entre procesos via TCP y UDP
BUFFER_SIZE = 1024  # Tamanio del buffer de comunicacion

i=raw_input()
data = read_csv('wine'+i+'.csv') # Lectura de los datos parcial de un conjunto de datos
data = DataFrame(scale(data), index=data.index, columns=data.columns)
XMat = data.rename_axis('ID').values # Se convierten los datos en una matriz.
XMat=XMat.T # Se transpone la matriz para ser consistente con el paper.
currentR=7 # Estimacion inicial del rango
currentR,currentU,currentS=SAPCA(currentR,XMat,125,.001,.5) # Se calculan las direcciones principales con los datos disponibles

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Se crea un socket para establecer comunicaciones cliente a cliente
s.bind((TCP_IP, 0))
puertoLider = s.getsockname()[1] # Se guarda el puerto del socket para comunicacion multicast
s.settimeout(2) # Se define el tiempo por el cual el proceso buscara ser lider.
# Despues de este tiempo el proceso dormira y entrara en modo participante.

while 1:
    # Se guardan las estimaciones actuales de U y S
    np.save('currentU', currentU)
    np.save('currentS', currentS)
    print("Soy Lider")
    time.sleep(1)
    send("lider "+str(puertoLider)) # Se le manda una invitacion a todos los participantes.
    try:
        s.listen(1)
        conn, addr = s.accept()
        # Si el cliente recibe una conexion significa que un participante le enviara su estimacion
        # actual de U y S por lo que se hara un merge.
        mensajeReconstruido = ""
        while 1:
            time.sleep(.1)
            data = conn.recv(BUFFER_SIZE)
            if not data: break
            mensajeReconstruido+=data
        #conn.close() # se cierra la conexion
        # Se decodificaran las matrices obtenidas del mensaje
        matrices = mensajeReconstruido.split('/') 
        incomingU=str2array(matrices[0])
        incomingS=str2array(matrices[1])
        incomingR=int(matrices[2])
        # Se hace un merge con las nuevas matrices.
        currentU,currentS=merge(max(currentR,incomingR),currentU,currentS,incomingU,incomingS)
        currentR = max(incomingR,currentR)
    except socket.timeout:
        # Si no se han obtenido respuestas el hilo entra en modo participante.
        soyParticipante=True
        q=list()
        thread1 = threading.Thread(target = Participante, args = (currentU,currentS,currentR, q))
        thread1.start()
        # Duerme un tiempo aleatorio para evitar deadlocks.
        time.sleep(random.randint(5,10))
        soyParticipante=False
        # Si el hilo sigue enviando algo permite que se termine el envio antes de resumir
        # en modo lider.
        while(thread1.isAlive()):
            time.sleep(1)
        if q.pop():
            break
    except Exception as e: 
        print("Excepcion de lider",e)
