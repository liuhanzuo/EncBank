"""Common action transport parser; AppWorld remains the execution sandbox."""
import re

VERSION = 'appworld-python-transport-v2'

def parse_action(text):
    # The native model may emit its Python tool transport despite the fenced-code prompt.
    if '<tool_call>' in text or '<function=' in text:
        calls = re.findall(r'<tool_call>\s*<function=python>\s*(.*?)\s*</function>\s*</tool_call>', text, re.S)
        if not calls or len(calls) != text.count('<tool_call>') or len(calls) != text.count('<function='):
            raise ValueError('Only complete native python tool calls are accepted')
        for call in calls:
            if '<tool_call>' in call or '<function=' in call or '<parameter=' in call:
                raise ValueError('Nested or unsupported Python tool encoding')
        return '\n\n'.join(calls), 'native_python_tool'
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    if blocks:
        return '\n'.join(blocks), 'python_fence'
    return text.strip(), 'plain_python'
