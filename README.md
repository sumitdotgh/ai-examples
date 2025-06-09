# :brain: ai-examples

A curated collection of practical AI and LLM examples using LangChain,OpenAI, Azure OpenAI, and LangGraph. This repository demonstrates how to build intelligent agents, integrate custom tools, work with APIs, and run local models like Mistral using Ollama.

## List of examples

| Name                    | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| Local LLM               | Run local LLM using Ollama                                         |
| basic-agent             | Basic calculation agent using LangChain and tool                   |
| basic-agent-langgraph   | Basic calculation agent using LangGraph, structured tool, and logging |


## Clean up script

```sh
docker rm -f $(docker ps -aq)
```