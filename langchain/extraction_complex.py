from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import warnings
from dotenv import load_dotenv
from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field

warnings.filterwarnings("ignore")
load_dotenv()

urls = ["https://en.wikipedia.org/wiki/Albert_Einstein"]

loader = UnstructuredURLLoader(urls=urls)

text = loader.load()

text = text[0].page_content


class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(default=None, description="The name of the person")
    date_of_birth: Optional[str] = Field(default=None, description="The date of birth of the person")
    scientific_papers: List[Optional[str]] = Field(default=None, description="List of scientific papers "
                                                                             "written by the person")


llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

structured_llm = llm.with_structured_output(
    Person,
    method="json_mode",
    include_raw=True
)

prompt = ("""Answer the following question. Make sure to return a JSON blob with keys 'answer' and 
'justification'.\n\n""" + text)

response = structured_llm.invoke(prompt)

print(response)
