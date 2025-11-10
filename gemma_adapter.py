"""
DSPy Adapter for Gemma models that don't support system prompts.

This adapter enables using Gemma models with DSPy by converting signatures
into user-only prompts and handling multimodal inputs (text + images).
"""

import json
import re
from dspy.adapters import Adapter


class GemmaAdapter(Adapter):
    """
    Custom DSPy adapter for Gemma models that don't support system prompts.

    This adapter:
    - Embeds all instructions (signature docstring, field descriptions) into the user prompt
    - Avoids using system messages entirely
    - Supports multimodal inputs (text and images)
    - Supports JSON and text response parsing

    Usage:
        import dspy
        from gemma_adapter import GemmaAdapter

        lm = dspy.LM(model="gemini/gemma-3-27b-it", api_key="YOUR_API_KEY")
        dspy.settings.configure(lm=lm, adapter=GemmaAdapter())

        # Now use DSPy normally with Signatures and Predict
    """

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        """
        Convert a DSPy signature into a user-only prompt and parse the response.

        Args:
            lm: Language model instance
            lm_kwargs: Additional LM parameters
            signature: DSPy Signature defining inputs and outputs
            demos: Example demonstrations (not used in this simple implementation)
            inputs: Actual input values to process

        Returns:
            List containing a dict of parsed outputs
        """
        # Build a single user prompt with all instructions embedded
        prompt_parts = []

        # Add signature docstring as task instructions
        if signature.__doc__:
            prompt_parts.append(signature.__doc__.strip())

        # Extract and format field descriptions
        input_fields = []
        output_fields = []

        for name, field in signature.input_fields.items():
            desc = field.json_schema_extra.get('desc', '') if field.json_schema_extra else ''
            input_fields.append(f"- {name}: {desc}")

        for name, field in signature.output_fields.items():
            desc = field.json_schema_extra.get('desc', '') if field.json_schema_extra else ''
            output_fields.append(f"- {name}: {desc}")

        if input_fields:
            prompt_parts.append("\nInputs:")
            prompt_parts.extend(input_fields)

        if output_fields:
            prompt_parts.append("\nOutputs:")
            prompt_parts.extend(output_fields)

        # Request specific output format
        prompt_parts.append("\nPlease provide the outputs in JSON format with these fields:")
        for name in signature.output_fields.keys():
            prompt_parts.append(f"- {name}")

        # Build content array with text and images
        content = []

        # Add the text prompt
        content.append({
            "type": "text",
            "text": "\n".join(prompt_parts)
        })

        # Add images if present in inputs
        for name, value in inputs.items():
            # Check if this looks like an image URL
            if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                # Add as image URL
                content.append({
                    "type": "image_url",
                    "image_url": {"url": value}
                })

        # Create a single user message with multimodal content
        messages = [{"role": "user", "content": content}]

        # Call the language model
        response = lm(messages=messages, **lm_kwargs)

        # Parse the response
        return self._parse_response(response, signature)

    def _parse_response(self, response, signature):
        """
        Parse the LM response, supporting both JSON and text formats.

        Args:
            response: Raw response from the language model
            signature: DSPy Signature to extract field names

        Returns:
            List containing a dict of parsed outputs
        """
        outputs = {}

        # Extract text from response
        if isinstance(response, list):
            response_text = response[0] if len(response) > 0 else ""
        else:
            response_text = str(response)

        # Try to parse JSON from code blocks (e.g., ```json {...} ```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                outputs = json.loads(json_match.group(1))
                return [outputs]
            except json.JSONDecodeError:
                pass

        # Try to parse the entire response as JSON
        try:
            outputs = json.loads(response_text)
            return [outputs]
        except json.JSONDecodeError:
            pass

        # Fallback: Simple text parsing for "field: value" format
        for line in response_text.strip().split('\n'):
            line = line.strip()
            for field_name in signature.output_fields.keys():
                if line.startswith(f"{field_name}:"):
                    value = line.split(':', 1)[1].strip()
                    outputs[field_name] = value

        return [outputs]
