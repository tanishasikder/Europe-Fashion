import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
from rq import Retry
import torch
from redis import Redis
load_dotenv()

connect = Redis.from_url(os.getenv('REDIS_URL'))
# Add as many queues for the heavy tasks
image_queue = Queue("default", connection=connect)
stats_queue = Queue("default", connection=connect)

# After user uploads the file, validate it, then put it here
# Wait for it to preprocess then put it in the model
def queue_image(img: torch.Tensor):
    image_queue.enqueue(img) # Put these in the queue

def queue_stats(): # Figure this out
    stats_queue.enqueue()