from inspect_ai import Task, task
from inspect_ai import dataset
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import Solver, generate


@task
def hello():
    return Task(
        dataset=[
            Sample(input="What is the Capital of France", target="Paris"),
            Sample(input="What is 2+2", target="4"),
        ],
        solver=generate(),
        scorer=includes(),
    )
