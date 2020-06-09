from tramp.algos.metrics import METRICS
import pandas as pd
import numpy as np
from tramp.base import ReprMixin
import logging
logger = logging.getLogger(__name__)
from tramp.algos.callbacks import Callback

class EarlyStopping_Custom(Callback):
    def __init__(self, ids="all", tol=1e-6, min_variance=-1, min_iterations=0,
                 wait_increase=5, max_increase=0.2):
        self.ids = ids
        self.tol = tol
        self.min_iterations = min_iterations
        self.min_variance = min_variance
        self.wait_increase = wait_increase
        self.max_increase = max_increase
        self.repr_init()
        self.old_vs = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.old_vs = None
        variables_data = algo.get_variables_data(self.ids)
        new_vs = [data["v"] for variable_id, data in variables_data.items()]
        if any(v < self.min_variance for v in new_vs):
            logger.info(f"early stopping min variance {min(new_vs)}")
            return True
        if any(np.isnan(v) for v in new_vs):
            logger.warning("early stopping nan values")
            logger.info("restoring old message dag")
            algo.reset_message_dag(self.old_message_dag)
            return True
        if self.old_vs:
            tols = [
                np.abs(old_v - new_v)
                for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            if max(tols) < self.tol and i >= self.min_iterations:
                logger.info(
                    "early stopping all tolerances (on v) are "
                    f"below tol={self.tol:.2e}"
                )
                return True
            increase = [
                new_v - old_v for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            if i > self.wait_increase and max(increase) > self.max_increase:
                logger.info(
                    f"increase={max(increase)} above "
                    f"max_increase={self.max_increase:.2e}"
                )
                logger.info("restoring old message dag")
                algo.reset_message_dag(self.old_message_dag)
                return True
        # for next iteration
        self.old_vs = new_vs
        self.old_message_dag = algo.message_dag.copy()