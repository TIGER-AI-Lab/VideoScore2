"""From https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py."""
"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
import json
from typing import Any
from datetime import datetime
import aiolimiter
import openai
from openai import AsyncOpenAI,OpenAI
from tqdm.asyncio import tqdm_asyncio
from zeno_build.models import lm_config

        
NOISE_CHARS="#.*: \n"


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    num_samples:int=None,
    logger=None,
) -> dict[str, Any]:
    if os.environ.get("OPENAI_ORG") is not None:
        client = AsyncOpenAI(organization=os.environ["OPENAI_ORG"],api_key=os.environ["OPENAI_API_KEY"])
    else:
        client = AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"])
    
    async with limiter:
        for _ in range(3):
            try:
                if num_samples is not None:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=num_samples,
                        top_p=top_p,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                if logger is not None:
                    logger.info(f"\n{response.usage}")
                else:
                    print(f"\n{response.usage}")
                await client.close()
                return response.to_dict()
               
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(5)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(5)
            except openai.BadRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    full_contexts: list[dict],
    model_config: lm_config.LMConfig,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    top_p: float = 1,
    requests_per_minute: int = 60,
    tqdm: bool = True,
    num_samples: int = 1,
    logger=None,
    
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_ORG") is None or os.environ.get("OPENAI_ORG")=="":
        openai.base_url=os.environ["OPENAI_BASE_URL"]
    else:
        openai.organization = os.environ["OPENAI_ORG"]
    
    
    if logger is not None:
        logger.info(f"model_name: {model_config.model}")
        logger.info(f"temperature: {temperature}")
        logger.info(f"max_tokens: {max_tokens}")
        logger.info(f"top_p: {top_p}")
        logger.info(f"requests_per_minute: {requests_per_minute}")
    else:
        print(f"model_name: {model_config.model}")
        print(f"temperature: {temperature}")
        print(f"max_tokens: {max_tokens}")
        print(f"top_p: {top_p}")
        print(f"requests_per_minute: {requests_per_minute}")
    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config.model,
            messages=full_context["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            logger=logger,
            num_samples=num_samples
        )
        for full_context in full_contexts
    ]
    if tqdm:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    
    if num_samples!=1:
        samples_2dlist=[]
        for response in responses:
            samples=[response["choices"][i]["message"]["content"] for i in range(len(response["choices"]))]
            samples_2dlist.append(samples)
        return samples_2dlist
    else:
        
        res=[response["choices"][0]["message"]["content"] for response in responses]
        # for idx, x in enumerate(res):
        #     if (len(x)) < 1000:
        #         print(f"xxxxxxxxxxxx {x}")
        #         print(f"bad index {res.index(x)}")
        return res





### -----------------------------------------------------------------------------------
### preprocess before running pipeline
### -----------------------------------------------------------------------------------
def init_client(model_name,key_idx):
    client = None
    API_KEYS=json.load(open("./const/api_key.json","r"))
    if "gpt" in model_name:
        API_KEY=API_KEYS[f"OpenAI_API_KEY{key_idx}"]
        ORG_ID=API_KEYS["OpenAI_ORG_ID"]
        client = OpenAI(organization=ORG_ID,api_key=API_KEY)
    if model_name=="deepseek-coder" or model_name == "deepseek-chat":
        API_KEY=API_KEYS["DeepSeek_API_KEY"]
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client


def set_logger(logger_file="./logs/test.log"):
    now = datetime.now()
    date_time = now.strftime("%m-%d--%H-%M-%S")
    logger_file=logger_file.replace(".log",f"_{date_time}.log")
    os.makedirs(os.path.dirname(logger_file),exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler(logger_file,encoding='utf-8'), 
                        logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    class HttpxFilter(logging.Filter):
        def filter(self, record):
            return "httpx" not in record.name
    logger.addFilter(HttpxFilter())
    return logger