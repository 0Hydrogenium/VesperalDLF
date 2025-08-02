import os
from openai import OpenAI
import json


class LLMAPIClient:

    qwen_llm = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_URL"))

    @classmethod
    def execute_qwen3_embedding(cls, text):
        completion = cls.qwen_llm.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=1024,
            encoding_format="float"
        )

        try:
            result = json.loads(completion.model_dump_json())
        except Exception as e:
            print(e)
            return None

        return [x['embedding'] for x in result['data']]


