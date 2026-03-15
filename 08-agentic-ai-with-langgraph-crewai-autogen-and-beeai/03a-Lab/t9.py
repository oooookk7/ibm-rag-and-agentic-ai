import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.agents.types import AgentExecutionConfig
import os
from dotenv import load_dotenv

load_dotenv()

async def reasoning_enhanced_agent_example():
    llm = ChatModel.from_name(
        "openai:Qwen/Qwen3-Coder-Next:novita",
        ChatModelParameters(temperature=0),
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
        base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        timeout=120
    )

    # SAME SYSTEM PROMPT as previous examples
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""
    
    # RequirementAgent with reasoning + research capability
    reasoning_agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],  # Thinking + Research
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            # ConditionalRequirement(
            #     ThinkTool,
            #     force_at_step=1,  # Thinking required first
            #     force_after=Tool,  # Force reasoning after every tool call
            #     min_invocations=1,  # At least once
            #     max_invocations=5,  # Max number of invocations
            #     consecutive_allowed=False  # No repeated thinking
            # ),
            #ConditionalRequirement(WikipediaTool, max_invocations=2)
        ]
    )
    
    # SAME QUERY as previous examples
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""
    
    result = await reasoning_agent.run(ANALYSIS_QUERY, execution=AgentExecutionConfig(max_iterations=30, total_max_retries=10))
    print(f"\n🧠 Reasoning + Research Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await reasoning_enhanced_agent_example()

if __name__ == "__main__":
    asyncio.run(main())
