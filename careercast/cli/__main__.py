"""
Executable module entrypoint for careercast.cli.
Enables running the CLI via: python -m careercast.cli [args]
"""

import sys
from careercast.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
