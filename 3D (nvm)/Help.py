import socket
import time

def send_message(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 5000))
    client_socket.sendall(message.encode('utf-8'))
    client_socket.close()

if __name__ == "__main__":
    while True:
        # Example: Move the right arm to a position
        bone_name = "mixamorig:RightArm"
        x = 0.1
        y = 0.2
        z = 0.3
        message = f"{bone_name},{x},{y},{z}"
        send_message(message)
        time.sleep(1)  # Send a message every second
