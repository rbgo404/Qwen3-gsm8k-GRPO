import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import inferless                             

app = inferless.Cls(gpu="A100")              


@inferless.request
class Request(BaseModel):
    problem: str = Field(default="Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each. His next 2 customers buy 2 DVDs each. His last 3 customers don't buy any DVDs. How many DVDs did Billy sell on Tuesday?")
    system_prompt: Optional[str] = "You are given a problem. Think about the problem and provide your reasoning. Place it between reasoning_start and reasoning_end. Then, provide your solution between: <answer> ... </answer>"
@inferless.response
class Response(BaseModel):
    answer: str


class InferlessPythonModel:
    @app.load
    def initialize(self):
        repo_id = "rbgo/Qwen3-gsm8k-GRPO"
        self.tokenizer  = AutoTokenizer.from_pretrained(repo_id)
        self.llm        = LLM(model=repo_id)
        self.sampling   = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)

    @app.infer
    def infer(self, request: Request) -> Response:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user",   "content": request.problem},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        outputs = self.llm.generate(prompt, self.sampling)
        return Response(answer=outputs[0].outputs[0].text)

    def finalize(self):
        self.llm = None
