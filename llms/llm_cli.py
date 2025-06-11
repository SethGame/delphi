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

import os
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
                "content": (
                    "You are a Historian. "
                    "You are given a country name, corresponding location, and type of armed forces "
                    "(e.g. army, navy). "
                    "You are to generate an army group/navy squadron name for that country in its corresponding "
                    "language but romanized if possible. "
                    "No other text should be generated. "
                    "If there was a historical name, use that instead."
                )
            },
            {"role": "user", "content": user_input_prompt},
        ],
    )
    return completion


def main():
    token_provider = get_token_provider()
    llm_model_list = ["gpt-4o", "gpt-4.1", "o3"]

    # Clear output directory
    output_dir = "./output"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(output_dir)

    prompts_list = [
        "Carthage, North Africa, Army",
        "Roman Empire, Albania, Army",
        "Mongol Empire, Manchuria, Army"
    ]

    for model in llm_model_list:
        # Create a single file for each model
        filename = f"./output/hoi4_names_{model}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            for idx, prompt in enumerate(prompts_list, 1):
                print(f"Model: {model}, Prompt {idx}")
                completion = generate_completion(token_provider, model, prompt)
                response = completion.choices[0].message.content
                print(f"\n{response}\n")
                
                # Write prompt and response to file
                file.write(f"Prompt {idx}: {prompt}\n")
                file.write(f"Response: {response}\n")
                file.write("-" * 50 + "\n")  # Add separator between entries


if __name__ == "__main__":
    main()
