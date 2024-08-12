import asyncio
import math
from dataclasses import dataclass

import weave

@dataclass
class Point:
    x: float
    y: float


ds = [{"point": Point(x=i, y=i)} for i in range(5)]
print(ds)


weave.init("bug_weave_dict")

@weave.op
def distance_to_origin(point: Point) -> float:
    return math.sqrt(point.x**2 + point.y**2)

@weave.op
def in_unit_circle(model_output: float) -> bool:
    return {"unit_circle": model_output < 1}


evaluation = weave.Evaluation(dataset=ds, scorers=[in_unit_circle])
asyncio.run(evaluation.evaluate(distance_to_origin))