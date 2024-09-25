from typing import Any, NamedTuple, TypedDict

import ipdb
import numpy as np

from pncbf.utils.spline_utils import get_spline


class TimedData(NamedTuple):
    t: np.ndarray
    v: np.ndarray


class DataStream(TypedDict):
    times: np.ndarray
    data: dict[str, np.ndarray]


class DataStreams:
    def __init__(self, streams: dict[str, DataStream], convert: bool = True):
        self.first_timestamp = 0
        self.streams = streams

        # Convert all timestamps to offset from first timestamp in seconds.
        if convert:
            self.first_timestamp, self.streams = self._stamp_to_seconds(streams)

    @staticmethod
    def _stamp_to_seconds(streams: dict[str, DataStream]):
        # Get all the times
        times = {k: v["times"][0] for k, v in streams.items()}
        # Get the first timestamp
        first_timestamp = min(times.values())
        # Convert all timestamps to offset from first timestamp in seconds.
        for topic, stream in streams.items():
            offset_ns = stream["times"] - first_timestamp
            offset_s = offset_ns / 1e9
            streams[topic]["times"] = offset_s

        return first_timestamp, streams

    def rename(self, old_topic: str, new_topic: str):
        self.streams[new_topic] = self.streams.pop(old_topic)

    def start_from(self, start_s: float):
        for topic, stream in self.streams.items():
            # Find the first index where the timestamp is greater than start_s.
            idx = np.argmax(stream["times"] > start_s)
            # Slice the times and data.
            self.streams[topic]["times"] = stream["times"][idx:]
            self.streams[topic]["data"] = {k: v[idx:] for k, v in stream["data"].items()}

    def end_at(self, end_s: float):
        for topic, stream in self.streams.items():
            # Find the first index where the timestamp is greater than start_s.
            idx = np.argmax(stream["times"] > end_s)
            # Slice the times and data.
            self.streams[topic]["times"] = stream["times"][:idx]
            self.streams[topic]["data"] = {k: v[:idx] for k, v in stream["data"].items()}

    def respline(self, dt: float) -> "DataStreams":
        """Respline all streams to a common time base."""
        t0_max = max([stream["times"][0] for stream in self.streams.values()])
        tf_min = min([stream["times"][-1] for stream in self.streams.values()])

        n_ts = int((tf_min - t0_max) / dt)
        H_t: np.ndarray = np.arange(n_ts) * dt

        splined_streams: dict[str, DataStream] = {}

        for topic, stream in self.streams.items():
            data_dict: dict[str, np.ndarray] = {}
            T_t = stream["times"] - t0_max
            for name, T_data in stream["data"].items():
                if T_data.dtype in [np.float32, np.float64]:
                    T_t_max = T_t.max()
                    spl = get_spline(T_t / T_t_max, T_data, k=1, s=0.0)
                    H_data = spl(H_t / T_t_max)
                    data_dict[name] = H_data
                else:
                    data_dict[name] = T_data

            splined_streams[topic] = {"times": H_t + t0_max, "data": data_dict}

        return DataStreams(splined_streams, convert=False)

    def __call__(self, topic: str, key: str):
        times = self.streams[topic]["times"]
        data = self.streams[topic]["data"][key]
        return TimedData(times, data)


class RosbagStore:
    def __init__(self):
        self.streams: dict[str, DataStream] = {}

    def finalize(self):
        # Convert lists to np arrays.
        for topic, stream in self.streams.items():
            self.streams[topic]["times"] = np.array(stream["times"])
            for k, v in stream["data"].items():
                self.streams[topic]["data"][k] = np.array(v)

    @property
    def handlers(self):
        return {
            "geometry_msgs/msg/PoseStamped": self.handle_pose_stamped,
            "geometry_msgs/msg/TwistStamped": self.handle_twist_stamped,
        }

    def add_msg(self, topic: str, timestamp: float, msg_type: str, msg: Any):
        if msg_type not in self.handlers:
            return

        msg_dict = self.handlers[msg_type](msg)
        if topic not in self.streams:
            self.streams[topic] = {"times": [], "data": {k: [] for k in msg_dict.keys()}}

        self.streams[topic]["times"].append(timestamp)
        for k, v in msg_dict.items():
            self.streams[topic]["data"][k].append(v)

    def handle_pose_stamped(self, msg: Any) -> dict[str, np.ndarray]:
        stamp = msg.header.stamp
        pose = msg.pose
        pos, quat = pose.position, pose.orientation
        pos = np.array([pos.x, pos.y, pos.z])
        quat = np.array([quat.x, quat.y, quat.z, quat.w])
        return {"stamp": stamp, "pos": pos, "quat": quat}

    def handle_twist_stamped(self, msg: Any) -> dict[str, np.ndarray]:
        stamp = msg.header.stamp
        lin = msg.twist.linear
        lin = np.array([lin.x, lin.y, lin.z])
        return {"stamp": stamp, "lin": lin}
