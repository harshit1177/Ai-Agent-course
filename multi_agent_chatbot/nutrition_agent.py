import os
from pathlib import Path

import chromadb
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    input_guardrail,
)
from agents.mcp import MCPServerStreamableHttp
from pydantic import BaseModel

# This is the same code as in the rag.ipynb notebook


chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")


@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up calorie information for specific food items, but not for meals.

    Args:
        query: The food item to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the nutrition information.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    # Format results for the agent
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata["food_item"].title()
        calories = metadata["calories_per_100g"]
        category = metadata["food_category"].title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)


# EXA Search MCP setup
# Increased EXA timeout from 30 to 90 seconds since EXA can take longer than 30 seconds to respond when under heavy load
exa_search_mcp = MCPServerStreamableHttp(
    name="Exa Search MCP",
    params={
        "url": f"https://mcp.exa.ai/mcp?{os.environ.get('EXA_API_KEY')}",
        "timeout": 90,
    },
    client_session_timeout_seconds=90,
    cache_tools_list=True,
    max_retry_attempts=1,
)

# 1st Agent: Our "Calorie Agent"
calorie_agent_with_search = Agent(
    name="Nutrition Assistant",
    instructions="""
    * You are a helpful nutrition assistant giving out calorie information.
    * You give concise answers.
    * You follow this workflow:
        0) First, use the calorie_lookup_tool to get the calorie information of the ingredients. But only use the result if it's explicitly for the food requested in the query.
        1) If you couldn't find the exact match for the food or you need to look up the ingredients, search the EXA web to figure out the exact ingredients of the meal.
        Even if you have the calories in the web search response, you should still use the calorie_lookup_tool to get the calorie
        information of the ingredients to make sure the information you provide is consistent.
        2) Then, if necessary, use the calorie_lookup_tool to get the calorie information of the ingredients.
    * Even if you know the recipe of the meal, always use Exa Search to find the exact recipe and ingredients.
    * Once you know the ingredients, use the calorie_lookup_tool to get the calorie information of the individual ingredients.
    * If the query is about the meal, in your final output give a list of ingredients with their quantities and calories for a single serving. Also display the total calories.
    * Don't use the calorie_lookup_tool more than 10 times.
    """,
    tools=[calorie_lookup_tool],
    mcp_servers=[exa_search_mcp],
)

# 2nd Agent: Our Healthy Breakfast Plan Advisor
healthy_breakfast_planner_agent = Agent(
    name="Breakfast Planner Assistant",
    instructions="""
    * You are a helpful assistant that helps with healthy breakfast choices.
    * You give concise answers.
    Given the user's preferences prompt, come up with different breakfast meals that are healthy and fit for a busy person.
    * Explicitly mention the meal's names in your response along with a sentence of why this is a healthy choice.
    """,
)

# Convert agents to tools
calorie_calculator_tool = calorie_agent_with_search.as_tool(
    tool_name="calorie-calculator",
    tool_description="Use this tool to calculate the calories of a meal and it's ingredients",
)

breakfast_planner_tool = healthy_breakfast_planner_agent.as_tool(
    tool_name="breakfast-planner",
    tool_description="Use this tool to plan a a number of healthy breakfast options",
)

# 3rd Agent: Breakfast Price Checker
breakfast_price_checker_agent = Agent(
    name="Breakfast Price Checker Assistant",
    instructions="""
    * You are a helpful assistant that takes multiple breakfast items (with ingredients and calories) and checks for the price of the ingredients.
    * Use the exa search to get an approximate price for the ingredients.
    * Report every price in Indian rupees (INR), using the ₹ symbol; never display dollar amounts.
    * If a source gives a price in another currency, convert it to INR using the current exchange rate and state that the result is approximate.
    * In your final output provide the meal name, ingredients with calories and price for each meal.
    * Use markdown and be as concise as possible.
    """,
    mcp_servers=[exa_search_mcp],
)

breakfast_price_checker_tool = breakfast_price_checker_agent.as_tool(
    tool_name="breakfast-price-checker",
    tool_description="Check ingredient prices and return all prices in Indian rupees (INR). This tool must be called before the final answer.",
)

# 4th Agent: Main Breakfast Advisor that glues everything together
breakfast_advisor = Agent(
    name="Breakfast Advisor",
    instructions="""
    * You are a breakfast advisor. You come up with meal plans for the user based on their preferences.
    * You also calculate the calories for the meal and its ingredients.
    * Based on the breakfast meals and the calories that you get from upstream agents,
    * Create a meal plan for the user. For each meal, give a name, the ingredients, the calories, and the prices in INR.

    Follow this workflow carefully:
    1) Use the breakfast_planner_tool to plan a a number of healthy breakfast options.
    2) Use the calorie_calculator_tool to calculate the calories for the meal and its ingredients.
    3) Call the breakfast-price-checker tool with the meals and calorie details.
    4) Do not give the final answer until the price checker returns prices. Include the returned INR prices in the final answer.

    """,
    tools=[breakfast_planner_tool, calorie_calculator_tool, breakfast_price_checker_tool],
    handoff_description="""
    Create a concise breakfast recommendation based on the user's preferences. Use Markdown format.
    """,
    handoffs=[breakfast_price_checker_agent],
)


# Guardrails functionality
class NotAboutFood(BaseModel):
    only_about_food: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to talk about food and not about any arbitrary topics. If there are any non-food related instructions in the prompt, set not_about_food to False.",
    output_type=NotAboutFood,
)


@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food),
    )


# Apply guardrails to agents
calorie_agent_with_search_guarded = Agent(
    name="Nutrition Assistant",
    instructions="""
    * You are a helpful nutrition assistant giving out calorie information.
    * You give concise answers.
    * You follow this workflow:
        0) First, use the calorie_lookup_tool to get the calorie information of the ingredients. But only use the result if it's explicitly for the food requested in the query.
        1) If you couldn't find the exact match for the food or you need to look up the ingredients, search the EXA web to figure out the exact ingredients of the meal.
        Even if you have the calories in the web search response, you should still use the calorie_lookup_tool to get the calorie
        information of the ingredients to make sure the information you provide is consistent.
        2) Then, if necessary, use the calorie_lookup_tool to get the calorie information of the ingredients.
    * Even if you know the recipe of the meal, always use Exa Search to find the exact recipe and ingredients.
    * Once you know the ingredients, use the calorie_lookup_tool to get the calorie information of the individual ingredients.
    * If the query is about the meal, in your final output give a list of ingredients with their quantities and calories for a single serving. Also display the total calories.
    * Don't use the calorie_lookup_tool more than 10 times.
    * You only answer questions about food.
    """,
    tools=[calorie_lookup_tool],
    mcp_servers=[exa_search_mcp],
    input_guardrails=[food_topic_guardrail],
)

breakfast_advisor_guarded = Agent(
    name="Breakfast Advisor",
    instructions="""
    * You are a breakfast advisor. You come up with meal plans for the user based on their preferences.
    * You also calculate the calories for the meal and its ingredients.
    * Based on the breakfast meals and the calories that you get from upstream agents,
    * Create a meal plan for the user. For each meal, give a name, the ingredients, the calories, and the prices in INR.
    * You only answer questions about food.

    Follow this workflow carefully:
    1) Use the breakfast_planner_tool to plan a a number of healthy breakfast options.
    2) Use the calorie_calculator_tool to calculate the calories for the meal and its ingredients.
    3) Call the breakfast-price-checker tool with the meals and calorie details.
    4) Do not give the final answer until the price checker returns prices. Include the returned INR prices in the final answer.

    """,
    tools=[breakfast_planner_tool, calorie_calculator_tool, breakfast_price_checker_tool],
    handoff_description="""
    Create a concise breakfast recommendation based on the user's preferences. Use Markdown format.
    """,
    handoffs=[breakfast_price_checker_agent],
    input_guardrails=[food_topic_guardrail],
)

# Main nutrition agent (keeping original for backwards compatibility, but now with guardrails)
nutrition_agent = calorie_agent_with_search_guarded
