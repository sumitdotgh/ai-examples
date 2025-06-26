## Tracing agent call with LangSmith

### Enable LangSmith

- SignUp https://www.langchain.com/langsmith
- Create an API key and save it.
- Export following configuration with details.
- You could also set this as environment variable.

```shell
export LANGCHAIN_PROJECT="basic-agent-tracing-langsmith"
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
```

### Trace Output

- Don't need any special call to any of the tracing API.
- Every time you run a chain or agent, LangSmith will now automatically capture the trace.

![Alt text](./images/output.png)


