import logging
import time
import tkinter
from queue import Full, Queue, Empty
from threading import Thread, Event
 
import PIL
from PIL import ImageTk
import cv2

logger = logging.getLogger("VideoStream")
 
 
def setup_webcam_stream(src=0):
    cap = cv2.VideoCapture(src)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Camera dimensions: {width, height}")
    logger.info(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    grabbed, frame = cap.read()  # Read once to init
    if not grabbed:
        raise IOError("Cannot read video stream.")
    return cap, width, height
 
 
def video_stream_loop(video_stream: cv2.VideoCapture, queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            success, img = video_stream.read()
            # We need a timeout here to not get stuck when no images are retrieved from the queue
            queue.put(img, timeout=1)
        except Full:
            pass  # try again with a newer frame
 
 
def processing_loop(input_queue: Queue, output_queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            img = input_queue.get()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, ::-1]  # mirror
            time.sleep(0.01)  # simulate some processing time
            # We need a timeout here to not get stuck when no images are retrieved from the queue
            output_queue.put(img, timeout=1)
        except Full:
            pass  # try again with a newer frame
 
 
class App:
    def __init__(self, window, window_title, image_queue: Queue, image_dimensions: tuple):
        self.window = window
        self.window.title(window_title)
 
        self.image_queue = image_queue
 
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=image_dimensions[0], height=image_dimensions[1])
        self.canvas.pack()
 
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
 
        self.window.mainloop()
 
    def update(self):
        try:
            frame = self.image_queue.get(timeout=0.1)  # Timeout to not block this method forever
            self.photo = ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.window.after(self.delay, self.update)
        except Empty:
            pass  # try again next time
 
 
def main():
    stream, width, height = setup_webcam_stream(0)
    webcam_queue = Queue()
    processed_queue = Queue()
    stop_event = Event()
    window_name = "FPS Multi Threading"
 
    try:
        Thread(target=video_stream_loop, args=[stream, webcam_queue, stop_event]).start()
        Thread(target=processing_loop, args=[webcam_queue, processed_queue, stop_event]).start()
        App(tkinter.Tk(), window_name, processed_queue, (width, height))
    finally:
        stop_event.set()
 
    print(f"Webcam queue: {webcam_queue.qsize()}")
    print(f"Processed queue: {processed_queue.qsize()}")
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
