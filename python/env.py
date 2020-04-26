import sys, subprocess, time, socket
import gym

SERVER_ADDRESS = ('localhost', 34343)

class IO:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(SERVER_ADDRESS)

    def sendMessage(self, msg):
        self.sock.sendall(msg.encode(encoding='UTF-8'))

    def readMessage(self):
        return self.sock.recv(128).decode(encoding='UTF-8')

    def dispose(self):
        self.sock.close()


class CarEnv(gym.Env):
    def __init__(self):
        self.javaProc = None
        self.io = None

    def reset(self):
        if self.javaProc is not None: self.javaProc.kill()
        self.javaProc = subprocess.Popen('java -jar desktop-1.0.jar')
        time.sleep(2)
        if self.io is not None: self.io.dispose()
        self.io = IO()
        return self.io.readMessage()

    def dispose(self):
        self.javaProc.kill()
        self.io.dispose()

    def step(self, action):
        if self.javaProc is None or self.io is None:
            raise('You need to call reset() first')

        msg = str(action[0]) + " "  +str(action[1]) + "\n"
        self.io.sendMessage(msg)
        answ = self.io.readMessage()

        obs = [float(x) for x in answ.split(" ")[:-1]]
        return obs

if __name__ == '__main__':
    try:
        env = CarEnv()
        for j in range(1):
            env.reset()
            for i in range(500):
                print(env.step([1, 0]))
    finally:
        print('cleanup')
        env.dispose()