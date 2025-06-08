# ai-examples
AI Examples

## List of examples

| Name                      | Description            
|--------------------------------------------------------------------------------------
| Local LLM                 | Run local LLM using ollama
| basic-agent               | Basic calculation agent using langchain and tool.
| basic-agent-langgraph     | Basic calculation agent using langgraph, structured tool and logging.

## Clean up script

```sh
docker rm -f $(docker ps -aq)
```