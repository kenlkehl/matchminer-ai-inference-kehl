"""Explicit, lazy backup configuration shared by the two workup reviewers."""

import copy
from threading import Lock

from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.backends import remote_enabled
from matchminer_ai.llm.structured import EndpointError


class WorkupBackup:
    def __init__(self, config, threshold):
        if type(threshold) is not int or not 1 <= threshold <= 10:
            raise ValueError(
                "max_consecutive_failures must be an integer from 1 to 10."
            )
        if config is not None:
            if not isinstance(config, MMAIConfig):
                raise TypeError("backup_config must be an MMAIConfig instance or None.")
            if not remote_enabled(config):
                raise ValueError(
                    "The backup requires a configured remote LLM endpoint."
                )
        self.config = copy.deepcopy(config)
        self.threshold = threshold
        self.lock = Lock()
        self.resolved = None
        self.failed = False

    @property
    def enabled(self):
        return self.config is not None

    def get(self, factory):
        # Resolution can probe the endpoint. Do it only when a failed item needs
        # its explicitly configured backup, once even with parallel questions.
        with self.lock:
            check_cancelled()
            if self.failed:
                raise EndpointError(
                    "The configured backup endpoint could not be prepared."
                )
            if self.resolved is None:
                try:
                    self.resolved = factory(self.config)
                except (EndpointError, ValueError):
                    self.failed = True
                    raise EndpointError(
                        "The configured backup endpoint could not be prepared."
                    ) from None
            check_cancelled()
            return self.resolved


def backup_event(primary, backup, reason, threshold, **details):
    """Keep failure provenance useful without provider errors or patient text."""
    return {
        "primary_model": primary.model,
        "backup_model": backup.model,
        "reason": reason,
        "failure_threshold": threshold,
        **details,
    }
