# Documentation & Catalog Synchronization Rule

Whenever a new subproject is added, removed, or significantly altered in `ai-examples`, all accompanying catalog assets across the repository must be updated synchronously.

## 1. Root `README.md`
- Add an entry in the **`List of examples`** markdown table.
- Use a relevant emoji, bold project name, and concise description.
- Example:
  ```markdown
  | 🤖 **my-new-agent** | Autonomous agent that handles customer ticket categorization |
  ```

## 2. MkDocs Configuration (`mkdocs.yml`)
- Add an entry under the `nav:` block matching the new documentation file.
- Example:
  ```yaml
  nav:
    ...
    - My New Agent: my-new-agent.md
  ```

## 3. Dedicated Documentation (`docs/<project-name>.md`)
- Create a markdown guide in `docs/` containing:
  - High-level overview and problem statement.
  - Architecture diagram using Mermaid (`graph TD` or `sequenceDiagram`).
  - Directory structure tree.
  - Prerequisites, setup steps, and sample run commands with expected output.

## 4. GitHub Actions CI Matrix (`.github/workflows/ci.yml`)
- For projects suitable for automated linting or testing, append the project name to the `matrix.project` list in `.github/workflows/ci.yml` so automated CI verifies it on push.
