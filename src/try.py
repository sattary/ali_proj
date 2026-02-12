"""
Compatibility shim for the legacy training entrypoint.

The original implementation has been refactored into the `phase_unwrap`
package. This script delegates to the Typer-based CLI so that existing
commands like `python src/try.py ...` continue to function, now backed
by the modular framework.
"""

from phase_unwrap.cli import main


if __name__ == "__main__":
    main()

