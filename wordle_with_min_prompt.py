from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.messages import HumanMessage
import os
import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("open ai api key here")

llm = ChatOpenAI(
    model = "gpt-3.5-turbo-0125",
    temperature = 0.1,
    max_tokens = 512
)

response = llm.invoke("Hi! I'm Bob")
print(response)