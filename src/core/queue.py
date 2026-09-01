import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
from rq.retry import Retry
import torch

load_dotenv()

redis = os.getenv('REDIS_URL')

# Add as many queues for the heavy tasks
image_queue = Queue("default", connection=redis)
stats_queue = Queue("default", connection=redis)

# After user uploads the file, validate it, then put it here
# Wait for it to preprocess then put it in the model
def queue_image(img: torch.Tensor):
    job = image_queue.enqueue(img)

def queue_stats(): # Figure this out
    job = stats_queue.enqueue()