import os
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import httpx

load_dotenv()
load_dotenv(".env.local", override=True)

model = OpenAIModel(
    os.getenv("MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = Agent(
    model,
    system_prompt='Be concise, reply with one sentence.',
)

# Register the function tool with the agent
@agent.tool_plain
def get_current_datetime() -> str:
    """Returns the current date and time in ISO format."""
    return datetime.now().isoformat()

@dataclass
class MyDeps:
    http_client: httpx.AsyncClient

@agent.tool
async def get_github_user_info(ctx: RunContext[MyDeps], username: str) -> str:
    """Fetches GitHub user information for the given username."""
    response = await ctx.deps.http_client.get(f"https://api.github.com/users/{username}")
    response.raise_for_status()
    return response.json()


async def main():
    # result = await agent.run('Where does "hello world" come from?')
    # print(result.data)

    # result = await agent.run('What is the current date and time?')
    # print(result.data)

    deps = MyDeps(http_client=httpx.AsyncClient())
    result = await agent.run('Tell me about the GitHub user "prakashrx"', deps=deps)
    print(result.data)


asyncio.run(main())