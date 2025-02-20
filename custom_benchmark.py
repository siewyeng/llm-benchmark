import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
from datetime import datetime


class Message(BaseModel):
    role: str
    content: str

models = ollama.list().get("models", [])
# print("list of models", models)

class EditedOllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

def nanosec_to_sec(nanosec):
    return nanosec / 1000000000
def inference_stats(model_response):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
        model_response.prompt_eval_count + model_response.eval_count
    ) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )

def average_stats(responses: list):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = EditedOllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)

def run_benchmark(
    model_name: str, prompt: str, verbose: bool
):

    last_element = None

    if verbose:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None
    
    # print("last element", last_element)
    return last_element


model_names = [model["model"] for model in models]
benchmarks = {}

prompts=[
            "Why is the sky blue?",
            "Write a report on the financials of Apple Inc.",
        ]
for model_name in model_names:
    # responses: List[OllamaResponse] = []
    responses = []
    for prompt in prompts:

        response = run_benchmark(model_name, prompt, verbose=False)
        responses.append(response)

    benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        # print("RESPONSES")
        # print(responses)
        average_stats(responses)

# for model_name, responses in benchmarks.items():
#     average_stats(responses)
