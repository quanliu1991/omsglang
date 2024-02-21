
from api.schemas.response import Answer
import sglang as sgl
from logging import getLogger
logger = getLogger()
from sglang import  set_default_backend, RuntimeEndpoint


@sgl.function
def image_qa(s, initial_prompt, image_file, question, max_tokens, gen_ignore_eos=False, fixed_length=(None, None)):
    s += sgl.system(initial_prompt)
    if image_file is None:
        s += sgl.user(question)
    else:
        s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens, ignore_eos=gen_ignore_eos, fixed_length=fixed_length))


class SGLangEngine:
    def __init__(self):
        pass

    async def batch_predict(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            parallel=1,
            ignore_eos=False
    ):
        # get backend
        # backend = Runtime(model_path=model_id,tokenizer_path="/app/llm_models/llava-1.5-7b-hf")
        # sgl.set_default_backend(backend)
        # set_default_backend(backend)
        backend = RuntimeEndpoint(f"http://127.0.0.1:30000")
        set_default_backend(backend)

        # prepara inputs
        arguments = [
            {"initial_prompt": initial_prompt,
             "image_file": prompt.image,
             "question": prompt.records[0].user,
             "max_tokens": max_tokens,
             "ignore_eos": ignore_eos} for prompt in prompts
        ]
        states = [None] * len(arguments)

        # run
        if parallel == 1:
            for i in range(len(arguments)):
                image_file = arguments[i]["image_file"]
                question = arguments[i]["question"]
                ret = image_qa.run(
                    initial_prompt=initial_prompt,
                    image_file=image_file,
                    question=question,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                    temperature=temperature)
                states[i] = ret
        else:
            states = image_qa.run_batch(
                arguments,
                max_new_tokens=max_tokens,
                ignore_eos=True,
                temperature=temperature,
                num_threads=parallel,
                progress_bar=True)

        # Write results
        generated_texts = []
        for state in states:
            for m in state.messages():
                if m["role"] == "assistant":
                    text = m["content"]
            input_tokens = state.stream_executor.meta_info["answer"]["prompt_tokens"]
            output_tokens = max_tokens
            mean_prob = None
            probs = {}
            generated_texts.append(
                Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens, mean_prob=mean_prob,
                       probs=probs))
        logger.info(generated_texts)
        return generated_texts

    async def profromance_banchmark(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            fixed_length=(None, None),
            parallel=1,
            ignore_eos=False
    ):
        # get backend
        # backend = Runtime(model_path=model_id,tokenizer_path="/app/llm_models/llava-1.5-7b-hf")
        # sgl.set_default_backend(backend)
        # set_default_backend(backend)
        backend = RuntimeEndpoint(f"http://127.0.0.1:30000")
        set_default_backend(backend)

        # prepara inputs
        arguments = []
        for prompt in prompts:
            arguments.append({"initial_prompt": initial_prompt,
                              "image_file": prompt.image,
                              "question": prompt.records[0].user,
                              "max_tokens": fixed_length[1],
                              "ignore_eos": ignore_eos,
                              "fixed_length": fixed_length
                              })

        return await self.run_infer(arguments, fixed_length, initial_prompt, max_tokens, parallel, temperature)

    async def run_infer(self, arguments, fixed_length, initial_prompt, max_tokens, parallel, temperature):
        states = [None] * len(arguments)
        # run
        if parallel == 1:
            for i in range(len(arguments)):
                image_file = arguments[i]["image_file"]
                question = arguments[i]["question"]
                ret = image_qa.run(
                    initial_prompt=initial_prompt,
                    image_file=image_file,
                    question=question,
                    max_tokens=fixed_length[1],
                    gen_ignore_eos=True,
                    ignore_eos=True,
                    fixed_length=fixed_length,
                    temperature=temperature)
                states[i] = ret
        else:
            states = image_qa.run_batch(
                arguments,
                max_new_tokens=fixed_length[1],
                temperature=temperature,
                num_threads=parallel,
                progress_bar=True)
        # Write results
        generated_texts = []
        for state in states:
            for m in state.messages():
                if m["role"] == "assistant":
                    text = m["content"]
            input_tokens = state.stream_executor.meta_info["answer"]["prompt_tokens"]
            output_tokens = fixed_length[1]
            mean_prob = None
            probs = {}
            generated_texts.append(
                Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens, mean_prob=mean_prob,
                       probs=probs))
        logger.info(generated_texts)
        return generated_texts


if __name__ == '__main__':
    import multiprocessing as mp

    mp.set_start_method("spawn")


    @sgl.function
    def image_qa(s, initial_prompt, image_file, question, max_tokens):
        s += sgl.system(initial_prompt)
        s += sgl.user(sgl.image(image_file) + question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens))


    # backend =Runtime(model_path="/app/llm_models/omchat-llava-vicuna-7b-v1.5-v1-1-finetune_zh_n92", tokenizer_path="/app/llm_models/llava-1.5-7b-hf" )
    backend = RuntimeEndpoint(f"http://127.0.0.1:30000")
    set_default_backend(backend)
    arguments = [
                    {"initial_prompt": "lalal",
                     "image_file": "https://minio.linker.cc/e-agent/agent/128/202401/31/170106027a6965245344c9a331645e9b5125a8.jpg",
                     "question": "你是一个监控机器人,你现在看到一个场景, 假设自己为该场景的管理人员，需要仔细查看图中的一切细节, 关注任何可能影响体验的细节",
                     "max_tokens": 1024}
                ] * 2

    ret = image_qa.run(
        initial_prompt="lalal",
        image_file=arguments[0]["image_file"],
        question=arguments[0]["question"],
        max_tokens=1024,
        temperature=0)
    # ret = image_qa.run_batch(
    #     arguments,
    #     temperature=0.1,
    #     num_threads=2,
    #     progress_bar=True)

    print(ret)
