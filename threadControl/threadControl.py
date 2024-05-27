import threading
import queue

class ThreadControl:
    def __init__(self, robot):
        self.threads = []
        self.checker_free = threading.Event()
        self.image_queue = queue.Queue()
        self.robot = robot
        self.exit_event = threading.Event()

