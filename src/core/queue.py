import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv

load_dotenv()

redis = os.getenv('REDIS_URL')

# Add as many queues for the heavy tasks
image_queue = Queue("default", connection=redis)
stats_queue = Queue("default", connection=redis)