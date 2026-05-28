import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from graph import app

# Generate Mermaid syntax
mermaid_code = app.get_graph().draw_mermaid()
print(mermaid_code)