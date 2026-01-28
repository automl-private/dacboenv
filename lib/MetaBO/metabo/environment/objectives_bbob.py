import numpy as np
import ioh
from pathlib import Path
import json

class BBOBFunction:
    """
    Base BBOB objective (maximization).
    Wraps IOH so the problem is created once.
    """

    def __init__(self, fid, dimension, instance=0, maximize=True, log_path=None, T=None):
        self.fid = fid
        self.dimension = dimension
        self.instance = instance
        self.problem = ioh.get_problem(
            fid=fid,
            dimension=dimension,
            instance=instance
        )
        self.sign = -1.0 if maximize else 1.0
        self.log_path = log_path
        self.T = T

    def __call__(self, x):
        x = np.asarray(x).reshape(-1, self.dimension)
        
        # Dump config to JSONL
        if self.log_path:
            json_path = Path(self.log_path) / "configs.jsonl"
            json_path.parent.mkdir(parents=True, exist_ok=True)

            if json_path.exists():
                with open(json_path, "r") as f:
                    t = sum(1 for _ in f)
            else:
                t = 0

            if self.T is not None:
                mod_t = t % self.T

            with open(json_path, "a") as f:
                f.write(
                    json.dumps({
                        "episode": t // self.T + 1,
                        "timestep": mod_t + 1,
                        "config": x.tolist(),
                    }) + "\n"
                )
        
        values = np.array(self.problem(x))
        return self.sign * values.reshape(-1, 1)


class TranslatedScaledBBOB:
    """
    Translated + scaled BBOB objective:
        f(x) = s * f_base(x - t)
    """

    def __init__(self, base_fn, bounds=(-5.0, 5.0)):
        self.base_fn = base_fn
        self.dim = base_fn.dimension
        self.lb, self.ub = bounds

    def __call__(self, x, t, s):
        x = np.asarray(x).reshape(-1, self.dim)
        t = np.asarray(t)

        # ensure translated points stay in domain
        t_min = self.lb - np.max(x, axis=0)
        t_max = self.ub - np.min(x, axis=0)
        t_clipped = np.clip(t, t_min, t_max)

        x_new = x - t_clipped
        return s * self.base_fn(x_new)

def bbob_max_min(fid, dimension, instance=0, maximize=True):
    """
    Global extrema of base BBOB function.
    """
    problem = ioh.get_problem(fid=fid, dimension=dimension, instance=instance)

    max_pos = np.array(problem.optimum.x).reshape(1, dimension)
    max_val = problem.optimum.y
    if maximize:
        max_val = -max_val

    # heuristic worst-case point (domain corner)
    min_pos = np.full((1, dimension), 5.0)
    min_val = problem(min_pos)[0]
    if maximize:
        min_val = -min_val

    return max_pos, max_val, min_pos, min_val


def bbob_max_min_var(fid, dimension, t, s, instance=0):
    """
    Extrema after translation and scaling.
    """
    max_pos, max_val, min_pos, min_val = bbob_max_min(
        fid, dimension, instance
    )

    t = np.asarray(t)

    t_min = -5.0 - max_pos.reshape(-1)
    t_max =  5.0 - max_pos.reshape(-1)
    t = np.clip(t, t_min, t_max)

    return (
        (max_pos.reshape(dimension,) + t).reshape(1, dimension),
        s * max_val,
        (min_pos.reshape(dimension,) + t).reshape(1, dimension),
        s * min_val
    )