"""
llm_cli.py

This script generates responses using either Azure OpenAI or Google Gemini models. It supports multiple LLM providers and can be configured via environment variables.

Main Components:
- get_azure_token_provider(): Returns a token provider for Azure OpenAI authentication.
- get_gemini_client(): Returns a configured Gemini client for Google's generative AI.
- generate_azure_completion(): Sends a prompt to Azure OpenAI and returns the completion.
- generate_gemini_completion(): Sends a prompt to Gemini and returns the completion.
- generate_completion(): Unified interface that routes to the appropriate provider.
- Main Execution: Iterates over LLM models and prompts, generates completions, and saves results.

Requirements:
- For Azure OpenAI: Azure OpenAI access and environment variables for Azure authentication
- For Gemini: Google Cloud API key (GEMINI_API_KEY environment variable)
- Required packages: openai, azure-identity, python-dotenv, google-generativeai

Usage:
Set LLM_PROVIDER environment variable to "azure_openai" or "gemini" (defaults to "gemini").
Run the script directly. Generated results will be saved in the output directory.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import AzureOpenAI
from azure.identity import EnvironmentCredential, get_bearer_token_provider


def get_azure_token_provider():
    """Get Azure OpenAI token provider for authentication."""
    load_dotenv()
    token_provider = get_bearer_token_provider(
        EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
    )
    print("Azure token_provider configured")
    return token_provider


def get_gemini_client():
    """Configure and return Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for Gemini API")
    
    genai.configure(api_key=api_key)
    print("Gemini client configured")
    return genai


def generate_azure_completion(token_provider, llm_model, user_input_prompt):
    """Generate completion using Azure OpenAI."""
    llm = AzureOpenAI(
        azure_endpoint="https://cog-sandbox-dev-eastus2-001.openai.azure.com/",
        api_version="2025-01-01-preview",
        azure_ad_token_provider=token_provider,
    )

    completion = llm.chat.completions.create(
        model=llm_model,
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
    return completion.choices[0].message.content


def generate_gemini_completion(client, llm_model, user_input_prompt):
    """Generate completion using Google Gemini."""
    model = client.GenerativeModel(llm_model)
    
    # Combine system and user prompts for Gemini
    full_prompt = (
        "You are a Historian. "
        "You are given a country name, corresponding location, and type of armed forces "
        "(e.g. army, navy). "
        "You are to generate an army group/navy squadron name for that country in its corresponding "
        "language but romanized if possible. "
        "No other text should be generated. "
        "If there was a historical name, use that instead.\n\n"
        f"Query: {user_input_prompt}"
    )
    
    response = model.generate_content(full_prompt)
    return response.text


def generate_completion(llm_model, user_input_prompt, provider="gemini"):
    """Unified completion generation interface supporting multiple providers."""
    if provider == "azure_openai":
        token_provider = get_azure_token_provider()
        return generate_azure_completion(token_provider, llm_model, user_input_prompt)
    elif provider == "gemini":
        client = get_gemini_client()
        return generate_gemini_completion(client, llm_model, user_input_prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def main():
    load_dotenv()
    
    # Get provider from environment variable, default to gemini
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    print(f"Using LLM provider: {provider}")
    
    # Model lists for different providers
    if provider == "azure_openai":
        llm_model_list = ["gpt-4o", "gpt-4.1", "o3"]
    elif provider == "gemini":
        llm_model_list = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")

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
        filename = f"./output/hoi4_names_{provider}_{model}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            for idx, prompt in enumerate(prompts_list, 1):
                print(f"Provider: {provider}, Model: {model}, Prompt {idx}")
                try:
                    response = generate_completion(model, prompt, provider)
                    print(f"\n{response}\n")
                    
                    # Write prompt and response to file
                    file.write(f"Prompt {idx}: {prompt}\n")
                    file.write(f"Response: {response}\n")
                    file.write("-" * 50 + "\n")  # Add separator between entries
                except Exception as e:
                    error_msg = f"Error generating completion: {str(e)}"
                    print(f"\n{error_msg}\n")
                    file.write(f"Prompt {idx}: {prompt}\n")
                    file.write(f"Error: {error_msg}\n")
                    file.write("-" * 50 + "\n")


if __name__ == "__main__":
    main()
