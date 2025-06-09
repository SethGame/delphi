from dotenv import load_dotenv
import os
from openai import AsyncAzureOpenAI
from azure.identity import EnvironmentCredential, get_bearer_token_provider
import chainlit as cl
from agents import set_default_openai_client, Agent, Runner, set_tracing_disabled
from mcp import ClientSession
from agents.mcp import MCPServer


class AppConfig:
    def __init__(self):
        load_dotenv()
        self.client = AsyncAzureOpenAI(
            azure_endpoint="https://cog-sandbox-dev-eastus2-001.openai.azure.com/",
            api_version="2025-03-01-preview",
            azure_ad_token_provider=get_bearer_token_provider(
                EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
            ),
        )
        set_default_openai_client(self.client)
        set_tracing_disabled(True)


async def agent_with_mcp(mcp_servers: MCPServer = None):
    if mcp_servers is None:
        mcp_servers = [
            MCPServer(
                name="openai",
                url="https://api.openai.com/v1/mcp/server",
            )
        ]

    agent = Agent(
        name="apollo",
        instructions="A helpful assistant that can answer questions",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        mcp_servers=mcp_servers,
    )
    return agent


# MCP
mcp_tools_cache = {}


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    mgs = cl.Message(content="")
    await mgs.send()

    agent = await agent_with_mcp()
    response = Runner.run_streamed(agent, message_history)

    async for event in response.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            token = event.data.delta
            await mgs.stream_token(token)
            await mgs.update()
        elif event.type == "tool_call_event":
            await mgs.stream_tool_calls(event.data.tool_calls)
            await mgs.update()
        elif event.type == "tool_result_event":
            await mgs.stream_tool_result(event.data.tool_result)
            await mgs.update()

    message_history.append({"role": "assistant", "content": mgs.content})
    cl.user_session.set("message_history", message_history)


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    cl.Message(f"Connected to MCP server: {connection.name}").send()

    try:
        result = await session.list_tools()

        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

        print(tools)

        mcp_tools_cache[connection.name] = tools

        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            f"Found {len(tools)} tools from {connection.name} MCP server."
        ).send()
    except Exception as e:
        await cl.Message(f"Error listing tools from MCP server: {str(e)}").send()


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    if name in mcp_tools_cache:
        del mcp_tools_cache[name]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    await cl.Message(f"Disconnected from MCP server: {name}").send()


if __name__ == "__main__":
    cl.run(AppConfig())
