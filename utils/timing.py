import time
from contextlib import contextmanager
from typing import Generator, Iterator

from utils.helpers import print_colored


@contextmanager
def measure_time(name: str) -> Generator[None, None, None]:
    """
    Context manager to measure and print execution time of a block.

    Args:
        name: Name of the operation being measured
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print_colored(f"{name}: {duration:.4f}s", "system")


def measure_generation(generator: Iterator[str]) -> Iterator[str]:
    """
    Wrapper for token generator to measure TTFT, total time, and TPS.

    Args:
        generator: The original token generator

    Yields:
        Tokens from the original generator
    """
    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        for token in generator:
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                print_colored(f"TTFT: {ttft:.4f}s", "system")

            token_count += 1
            yield token

    finally:
        end_time = time.time()
        total_time = end_time - start_time

        # Avoid division by zero
        if total_time > 0:
            tps = token_count / total_time
        else:
            tps = 0.0

        print_colored(f"\nTotal generation time: {total_time:.4f}s", "system")
        print_colored(f"TPS: {tps:.2f}", "system")
