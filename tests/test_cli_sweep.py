"""Check task construction, completion reporting, and scheduler event handling."""

import queue

import pytest

import lab_orchestrator
import main
from lab_orchestrator import scheduler
from lab_orchestrator.task import Task
from lab_orchestrator.worker import _physical_gpu_id
from stochastic_check.config import SweepConfig
from stochastic_check.sweep import build_sweep_tasks, completion_status


def test_task_commands_and_dependencies():
    """Keep comparison after its training task without changing command arguments."""
    config = SweepConfig(
        archs=["resnet18"], datasets=["cifar10"], noise_rates=[0.4], seeds=[3]
    )
    train, compare = build_sweep_tasks(config, [0, 10, 30], "both", False)
    assert train.name == "train:arch=resnet18_dataset=cifar10_noise=0.40_seed=3"
    assert compare.depends_on == [train.name]
    assert train.cmd[3:] == [
        "train",
        "--arch",
        "resnet18",
        "--dataset",
        "cifar10",
        "--noise",
        "0.4",
        "--seed",
        "3",
        "--epochs",
        "30",
        "--checkpoint-epochs",
        "0",
        "10",
        "30",
        "--output-dir",
        "results/exp_b/runs",
    ]

    assert compare.cmd[3:] == [
        "compare",
        "--arch",
        "resnet18",
        "--dataset",
        "cifar10",
        "--noise",
        "0.4",
        "--seed",
        "3",
        "--runs-root",
        "results/exp_b/runs",
        "--output-dir",
        "results/exp_a/runs",
    ]
    assert (train.weight, compare.weight) == (1.5, 0.75)


@pytest.mark.parametrize(
    "codes,labels,expected",
    [
        ([0, 0], ["one[seed=0]", "two[seed=0]"], 0),
        ([1, 0], ["one[seed=0]", "two[seed=0]"], 1),
        ([0], ["one[seed=0]"], 1),
        ([0, 0], ["one[seed=0]", "one[seed=0]"], 1),
        ([-2, 0], ["one[seed=0]", "two[seed=0]"], 1),
    ],
)
def test_completion_status(codes, labels, expected):
    """Require a successful result for every scheduled task exactly once."""
    tasks = [Task(1.0, "one", 0), Task(1.0, "two", 0)]
    results = [(0, label, code, 0.1) for label, code in zip(labels, codes, strict=True)]
    assert completion_status(tasks, results) == expected


@pytest.mark.parametrize("code", [0, 1])
def test_cli_propagates_schedule_status(monkeypatch, code):
    """Report task failure through the real CLI handler's exit status."""

    def schedule_result(**kwargs):
        """Replace process execution with matching task completion records."""
        return [(0, task.label, code, 0.1) for task in kwargs["tasks"]]

    monkeypatch.setattr(lab_orchestrator, "run_schedule", schedule_result)
    result = main.main(
        [
            "sweep",
            "--archs",
            "resnet18",
            "--datasets",
            "cifar10",
            "--noises",
            "0.4",
            "--seeds",
            "0",
            "--gpu-ids",
            "0",
        ]
    )
    assert result == code


@pytest.mark.parametrize(
    "gpu,visible,expected",
    [
        (0, None, "0"),
        (1, "2,5", "5"),
        (0, " GPU-a , GPU-b ", "GPU-a"),
        (4, "2,5", "4"),
    ],
)
def test_gpu_mapping_is_unchanged(gpu, visible, expected):
    """Interpret GPU IDs within the parent visibility mask when possible."""
    assert _physical_gpu_id(gpu, visible) == expected


class _EventQueue:
    """Replay scheduler events without starting or waiting for real workers."""

    def __init__(self, events=()):
        """Store incoming events and outgoing task assignments separately."""
        self.events = iter(events)
        self.sent = []

    def get(self, timeout=None):
        """Return the next planned event immediately."""
        return next(self.events)

    def get_nowait(self):
        """Keep each planned event in its own scheduler batch."""
        raise queue.Empty

    def put(self, value):
        """Record a task assignment or termination marker."""
        self.sent.append(value)


class _Worker:
    """Provide process identity/lifecycle without executing a subprocess."""

    pid = 1234

    def __init__(self, *args, **kwargs):
        """Accept the scheduler's normal process construction arguments."""

    def start(self):
        """Leave event delivery to the controlled test queue."""

    def is_alive(self):
        """Remain alive until the scheduler sends its termination marker."""
        return True

    def join(self, timeout=None):
        """Complete immediately without a real process to join."""


@pytest.mark.parametrize("failed", [False, True])
def test_scheduler_preserves_result_and_dependency_events(
    monkeypatch, tmp_path, failed
):
    """Retain successful order and cascade failures through dependent tasks."""
    tasks = [Task(1.0, "train", 0), Task(0.5, "compare", 0, depends_on=["train"])]
    code = 1 if failed else 0
    events = [
        ("READY", 0),
        ("START", 0, "train[seed=0]", 1.0, 1234),
        ("DONE", 0, "train[seed=0]", code, 0.25, 1.0, 1234),
    ]

    if not failed:
        events.extend(
            [
                ("READY", 0),
                ("START", 0, "compare[seed=0]", 0.5, 1234),
                ("DONE", 0, "compare[seed=0]", 0, 0.1, 0.5, 1234),
            ]
        )
    events.append(("READY", 0))

    incoming, outgoing = _EventQueue(events), _EventQueue()
    queues = iter([incoming, outgoing])
    monkeypatch.setattr(scheduler.mp, "Queue", lambda: next(queues))
    monkeypatch.setattr(scheduler.mp, "Process", _Worker)
    results = scheduler.run_schedule([0], tasks, log_dir=str(tmp_path))

    expected = [(0, "train[seed=0]", code, 0.25)]
    expected.append(
        (-1, "compare[seed=0]", -2, 0.0) if failed else (0, "compare[seed=0]", 0, 0.1)
    )
    assert results == expected
    assert outgoing.sent == ([tasks[0], None] if failed else [*tasks, None])
