import asyncio
from dataclasses import dataclass
from pathlib import Path

import weave
import simple_parsing

from mini_lib.utils import check_solution, load_jsonl

@dataclass
class Args(simple_parsing.Serializable):
    results_file: Path = Path("./results.jsonl")
    debug: bool = False # set to True to see the debug logs

if __name__=="__main__":
    args = simple_parsing.parse(Args)
    weave.init("hack-starter")
       
    dataset = load_jsonl(args.results_file)
    
    @weave.op
    def model(generated_output: Path):
        "dummy model passes the path"
        return generated_output

    @weave.op
    def match(model_output: str, output: str):
        matches = check_solution(
            Path(model_output).read_text(), 
            Path(output).read_text())
        return matches


    # run the eval
    evaluation = weave.Evaluation(dataset=dataset, scorers=[match])
    asyncio.run(evaluation.evaluate(model))