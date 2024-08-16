import warnings
from dotenv import load_dotenv
from enum import Enum
from typing import Union
from pydantic import BaseModel
import openai
from openai import OpenAI
import newspaper

warnings.filterwarnings("ignore")

load_dotenv()

def download_article(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    article_text = article.text
    return article_text

url = "https://techcrunch.com/2024/08/13/made-by-google-2024-a-few-ai-features-you-mightve-missed/"
article = download_article(url)

# Now, 'content' contains the text of the article
print("Article content:")
print(article[:500] + "...")

from enum import Enum
from pydantic import BaseModel

class ProductType(str, Enum):
    """
    Enumeration of product types.

    This enum represents different categories of products that can be referenced.
    """
    device = "device"
    app = "app"
    ft = "fintech"
    saas = "saas"
    consumer_tech = "consumer_tech"

class OrgType(str, Enum):
    """
    Enumeration of product types.

    This enum represents different categories of products that can be referenced.
    """
    startup = "startup"
    big_tech = "big_tech"
    tech_media = "tech_media"
    social_media = "social_media"
    gov = "govt_org"
    non_prof = "non_profit_org"
    vc = "venture_captial"
    other = "other"

class Person(BaseModel):
    """
    Represents an individual associated with an organization.

    This model captures basic information about a person, including their name,
    the organization they're associated with, and their role within that organization.
    """
    name: str
    organization: str
    role: str

class Product(BaseModel):
    """
    Represents a product offered by an organization.

    This model captures basic information about a product, including its name,
    the organization it belongs to, and its type.
    """
    name: str
    organization: str
    product_type: ProductType

class Organization(BaseModel):
    """
    Represents an organization or company.

    This model captures basic information about an organization, including its name
    and location.
    """
    name: str
    org_type: OrgType
    location: str

class ArticleResponse(BaseModel):
    """
    Represents the structured response for an article analysis.

    This model aggregates information about products, people, and organizations
    mentioned in an article, along with a summary of the article's content.
    """
    products: list[Product]
    people: list[Person]
    organizations: list[Organization]
    summary: str

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that scans for \
        people, products and organizations mentioned in articles."},
        {"role": "user", "content": article},
    ],
    response_format=ArticleResponse,
)

message = completion.choices[0].message
if message.parsed:
    print(message.parsed.people)
    for product in message.parsed.products:
        print(product)
    print(message.parsed.organizations)
    print(f"\n\nSummary: {message.parsed.summary}")
else:
    print(message.refusal)

