def extract_code_language(input_string: str) -> Tuple[str, Optional[str]]:
    """Extracts a programming language from the beginning of a string.

    This function checks if the input string starts with a pattern of the form
    ``<_some_language_>``. If it does, it extracts the language string and returns
    a tuple of (remainder, language). Otherwise, it returns the original string
    and `None`.

    Args:
        input_string (str): The input string, which may start with ``<_language_>``.

    Returns:
        Tuple[str, Optional[str]]:
            A tuple where:
            - The first element is either:
                - The remainder of the string (everything after ``<_language_>``),
                if a match is found; or
                - The original string, if no match is found.
            - The second element is the extracted language if a match is found;
            otherwise, `None`.
    """
    pattern = r"^<_([^>]+)_>\s*(.*)"
    match = re.match(pattern, input_string, flags=re.DOTALL)
    if match:
        language = str(match.group(1))  # the captured programming language
        remainder = str(match.group(2))  # everything after the <_language_>
        return remainder, language
    else:
        return input_string, None


def get_code_language_enum(value: Optional[str]) -> CodeLanguageLabel:
    """
    Converts a string to a corresponding `CodeLanguageLabel` enum member.

    If the provided string does not match any value in `CodeLanguageLabel`,
    it defaults to `CodeLanguageLabel.UNKNOWN`.

    Args:
        value (Optional[str]): The string representation of the code language or None.

    Returns:
        CodeLanguageLabel: The corresponding enum member if the value is valid,
        otherwise `CodeLanguageLabel.UNKNOWN`.
    """
    if not isinstance(value, str):
        return CodeLanguageLabel.UNKNOWN

    try:
        return CodeLanguageLabel(value)
    except ValueError:
        return CodeLanguageLabel.UNKNOWN
