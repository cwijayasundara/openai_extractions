import warnings
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
import json

warnings.filterwarnings("ignore")
load_dotenv()

llm = OpenAI(model="gpt-4o-2024-08-06")

response = llm.complete(
    "Generate a sales call transcript, use real names, talk about a product, discuss some action items"
)

transcript = response.text


class CallSummary(BaseModel):
    """Data model for a call summary."""

    summary: str = Field(
        description="High-level summary of the call transcript. Should not exceed 3 sentences."
    )
    products: List[str] = Field(
        description="List of products discussed in the call"
    )
    rep_name: str = Field(description="Name of the sales rep")
    prospect_name: str = Field(description="Name of the prospect")
    action_items: List[str] = Field(description="List of action items")


# Data extraction with JSON mode¶

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights from sales call transcripts.\n"
                "Generate a valid JSON following the given schema below:\n"
                "{json_schema}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ]
)

messages = prompt.format_messages(
    json_schema=CallSummary.schema_json(), transcript=transcript
)

output = llm.chat(
    messages, response_format={"type": "json_object"}
).message.content

print(output)

print("+++++++++++++++++++++++++++++++++++++++++++++++++")

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights from sales call transcripts.\n"
                "Generate a valid JSON in the following format:\n"
                "{json_example}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ]
)

dict_example = {
    "summary": "High-level summary of the call transcript. Should not exceed 3 sentences.",
    "products": ["product 1", "product 2"],
    "rep_name": "Name of the sales rep",
    "prospect_name": "Name of the prospect",
    "action_items": ["action item 1", "action item 2"],
}

json_example = json.dumps(dict_example)

messages = prompt.format_messages(
    json_example=json_example, transcript=transcript
)

output = llm.chat(
    messages, response_format={"type": "json_object"}
).message.content

print(output)
