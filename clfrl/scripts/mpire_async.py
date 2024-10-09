import multiprocessing as mp
import time
from ctypes import c_int

import ipdb
from mpire import WorkerPool


def worker_fn(wid: int, que: mp.Queue, xs: list[float]):
    print("wid: ", wid)
    for x in xs:
        que.put(1)
        time.sleep(1)
    return 0


def main():
    m = mp.Manager()
    que: mp.Queue = m.Queue()
    b_xs = [list(range(10)) for _ in range(2)]

    total = 0

    with WorkerPool(n_jobs=2, shared_objects=que, pass_worker_id=True, start_method="spawn") as pool:
        async_results = [pool.apply_async(worker_fn, args=(xs,)) for xs in b_xs]
        while not all(async_result.ready() for async_result in async_results):
            for _ in range(que.qsize()):
                total += que.get()
            print(total)
            time.sleep(0.5)
        results = [async_result.get() for async_result in async_results]


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
