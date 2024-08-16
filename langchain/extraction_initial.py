from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


class AnswerWithJustification(BaseModel):
    answer: str
    justification: str


llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

structured_llm = llm.with_structured_output(
    AnswerWithJustification,
    method="json_mode",
    include_raw=True
)

response = structured_llm.invoke(
    "Answer the following question. "
    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
    "What's heavier a pound of bricks or a pound of feathers?"
)

print(response)
