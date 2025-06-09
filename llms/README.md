## Azure OpenAI
https://platform.openai.com/docs/overview?lang=python

### About llm_cli.py
- This script demonstrates generating division names for various armies using Azure OpenAI models.
- It uses environment variable-based authentication, iterates through a list of prompts and models, prints results, and saves them to files.
- When executed, results for each model/prompt are saved in the output folder.
- Main functions:
    - get_token_provider(): Creates an authentication token provider
    - generate_completion(): Sends a prompt to the LLM and returns the result
    - main(): Manages the overall execution flow
- Before use, you must provide Azure authentication information in the .env file.

### What you need to do
- use Gemini 2.5 flash instead of Azure openai :) 
    - cursor will help you out