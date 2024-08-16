import warnings
import re
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import BSHTMLLoader
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

warnings.filterwarnings("ignore")
load_dotenv()

# Download the content
response = requests.get("https://en.wikipedia.org/wiki/Car")

with open("../docs/car.html", "w", encoding="utf-8") as f:
    f.write(response.text)

# Load it with an HTML parser
loader = BSHTMLLoader("../docs/car.html")
document = loader.load()[0]

# Replace consecutive new lines with a single new line
document.page_content = re.sub("\n\n+", "\n", document.page_content)

print(len(document.page_content))


class KeyDevelopment(BaseModel):
    """Information about a development in the history of cars."""
    year: int = Field(
        ..., description="The year when there was an important historic development."
    )
    description: str = Field(
        ..., description="What happened in this year? What was the development?"
    )
    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the year and description information were extracted",
    )


class ExtractionData(BaseModel):
    """Extracted information about key developments in the history of cars."""

    key_developments: List[KeyDevelopment]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found "
            "in the text.",
        ),
        ("human", "{text}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    # Remember to set the temperature to 0 for extractions!
    temperature=0,
)

extractor = prompt | llm.with_structured_output(
    ExtractionData,
    method="json_mode",
    include_raw=True
)


# Brute force approach

text_splitter = TokenTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
)

texts = text_splitter.split_text(document.page_content)

first_few = texts[:5]

extractions = extractor.batch(
    [{"text": text} for text in first_few],
    {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
)

# Merge results

key_developments = []

for extraction in extractions:
    key_developments.extend(extraction.key_developments)

# Print the results
for key_development in key_developments:
    print(key_development.year, key_development.description, key_development.evidence)