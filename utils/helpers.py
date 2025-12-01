from datetime import datetime

import yaml
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def print_colored(text: str, color: str = "white", prefix: str = "", end: str = "\n"):
    """
    Print colored text to console.

    Args:
        text: Text to print
        color: Color name (user, assistant, system, error)
        prefix: Optional prefix to add before text
        end: String appended after the last value, default is a newline.
    """
    color_map = {
        "user": Fore.CYAN,
        "assistant": Fore.GREEN,
        "system": Fore.YELLOW,
        "error": Fore.RED,
        "white": Fore.WHITE,
    }

    selected_color = color_map.get(color.lower(), Fore.WHITE)

    if prefix:
        print(
            f"{selected_color}{Style.BRIGHT}{prefix}{Style.RESET_ALL}{selected_color}{text}{Style.RESET_ALL}",
            end=end,
        )
    else:
        print(f"{selected_color}{text}{Style.RESET_ALL}", end=end)


def format_conversation_history(turns: list) -> str:
    """
    Format conversation history for display.

    Args:
        turns: List of conversation turns

    Returns:
        Formatted string representation
    """
    formatted = []
    for turn in turns:
        role = turn["role"].upper()
        content = turn["content"]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)
