import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 34343)
print('connecting to %s port %s' % server_address)
sock.connect(server_address)
print('connected')

while True:
    message = '1 1\n'
    print('sending')
    sock.sendall(message.encode(encoding='UTF-8'))

    data = sock.recv(64)
    print('received "%s"' % data.decode(encoding='UTF-8'))