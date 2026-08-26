"""
Run separately from the API process, e.g.:
    python worker.py
or scale out with several instances:
    python worker.py &  python worker.py &
 
Each worker pulls one job at a time off the "default" queue and runs it
to completion in its own process — CPU-heavy work here never touches
the FastAPI event loop.
"""
from rq import Worker
from src.app.core.queue import image_queue, stats_queue, redis

if __name__ == "__main__":
    img_worker = Worker([image_queue], connection=redis)
    stats_worker = Worker([stats_queue], connection=redis)

    img_worker.work()
    stats_worker.work()
