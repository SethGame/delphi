"""
fusion_scriptor.py

This script generates Python scripts using the Autodesk Fusion 360 API by leveraging Azure OpenAI models. It automates the creation of design scripts for various objects (e.g., tables, chairs, brackets, etc.) by sending prompts to different LLMs and saving the generated code to files.

Main Components:
- get_token_provider():
    Loads environment variables and returns a token provider for Azure OpenAI authentication.
- generate_completion(token_provider, llm_model, user_input_prompt):
    Sends a prompt to the specified LLM model and returns the generated completion.
- Main Execution:
    Iterates over a list of LLM models and prompts, generates completions, prints them, and saves each result to a uniquely named Python file.

Requirements:
- Azure OpenAI access
- Environment variables for Azure authentication (see .env file)
- openai, azure-identity, python-dotenv packages

Usage:
Run the script directly. Generated scripts will be saved in the current directory.
"""

from openai import AzureOpenAI
from azure.identity import EnvironmentCredential, get_bearer_token_provider


def get_token_provider():
    from dotenv import load_dotenv

    load_dotenv()
    token_provider = get_bearer_token_provider(
        EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
    )
    print("token_provider", token_provider)
    return token_provider


def generate_completion(token_provider, llm_model, user_input_prompt):
    llm = AzureOpenAI(
        azure_endpoint="https://cog-sandbox-dev-eastus2-001.openai.azure.com/",
        api_version="2025-01-01-preview",
        azure_ad_token_provider=token_provider,
    )

    completion = llm.chat.completions.create(
        model=llm_model,  # Use llm_model parameter
        messages=[
            {
                "role": "system",
                "content": "You are a historian. You are given a list of divisions of armies in WW2. You are to generate a name for each division.",
            },
            {
                "role": "user",
                "content": user_input_prompt
            }, 
        ],
    )
    return completion


def main():
    token_provider = get_token_provider()
    llm_model_list = ["gpt-4o", "gpt-4.1"]

    prompts_list = [
        "Generate name of division of Italian army in WW2",
        "Generate name of division of German army in WW2",
        "Generate name of division of Russian army in WW2",
        "Generate name of division of French army in WW2",
        "Generate name of division of British army in WW2",
        "Generate name of division of American army in WW2",
        "Generate name of division of Japanese army in WW2",
    ]

    for model in llm_model_list:
        for idx, prompt in enumerate(prompts_list, 1):
            print(f"Model: {model}, Prompt {idx}")
            completion = generate_completion(token_provider, model, prompt)
            print(f"\n{completion.choices[0].message.content}\n")

            filename = f"./output/division_name_{model}_prompt{idx}.txt"
            with open(filename, "w", encoding="utf-8") as file:
                file.write(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
