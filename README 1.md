# Gemma with DSPy - Pure DSPy Implementation

A pure DSPy implementation for using Gemma models, which don't support system prompts.

## Problem

Gemma models don't support system prompts (developer instructions), but DSPy's default adapters rely on them. This causes errors like:

```
BadRequestError: Developer instruction is not enabled for models/gemma-3-27b-it
```

## Solution

This project provides `GemmaAdapter`, a custom DSPy adapter that:
- Converts DSPy signatures into user-only prompts (no system messages)
- Supports multimodal inputs (text + images)
- Handles JSON and text response parsing
- Works with all standard DSPy patterns (Signature, Predict, Module)

## Files

- `gemma_adapter.py`: Reusable adapter for Gemma models
- `test_gemma_with_dspy.py`: Example usage with PAN card extraction
- `pyproject.toml`: Dependencies

## Quick Start

```python
import dspy
from gemma_adapter import GemmaAdapter

# Configure DSPy with Gemma
lm = dspy.LM(
    model="gemini/gemma-3-27b-it",
    api_key="YOUR_API_KEY"
)
dspy.settings.configure(lm=lm, adapter=GemmaAdapter())

# Define your task with standard DSPy Signature
class MyTask(dspy.Signature):
    """Task description here."""
    input_field: str = dspy.InputField(desc="Input description")
    output_field: str = dspy.OutputField(desc="Output description")

# Use standard DSPy Module
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MyTask)

    def forward(self, input_field: str):
        return self.predictor(input_field=input_field)

# Use it
module = MyModule()
result = module(input_field="test")
print(result.output_field)
```

## Features

### Multimodal Support (Images + Text)

The adapter automatically detects image URLs in inputs and sends them as images:

```python
class ImageTask(dspy.Signature):
    """Extract information from an image."""
    image_url: str = dspy.InputField(desc="URL of the image")
    description: str = dspy.OutputField(desc="Description of the image")

# The adapter will automatically send the image_url as an actual image
result = predictor(image_url="https://example.com/image.jpg")
```

### Response Parsing

The adapter supports multiple response formats:
1. JSON in code blocks: ` ```json {...} ``` `
2. Raw JSON: `{...}`
3. Text format: `field: value`

## Running the Example

```bash
# Install dependencies
uv sync

# Run the PAN card extraction example
uv run test_gemma_with_dspy.py
```

Expected output:
```
Testing Gemma with Pure DSPy (Custom Adapter)...
Name: MANOJ KUMAR PANDEY
Valid PAN: 1
```

## How It Works

1. **Signature Conversion**: The adapter takes a DSPy Signature and converts it into a user prompt by embedding:
   - The signature's docstring
   - Input field descriptions
   - Output field descriptions

2. **Multimodal Handling**: Image URLs are detected and sent as `image_url` content types

3. **Response Parsing**: The adapter tries multiple parsing strategies to extract structured outputs

## Use Cases

This adapter is useful for:
- Using Gemma models with DSPy
- Vision tasks with Gemma (document extraction, image analysis, etc.)
- Any model that doesn't support system prompts
- Multimodal tasks requiring both text and image inputs

## Dependencies

- `dspy-ai`: DSPy framework
- `litellm`: LLM provider abstraction (handles Gemini/Gemma)

## License

This is example code for educational purposes.
