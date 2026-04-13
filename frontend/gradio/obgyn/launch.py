"""Launch GynecologistGemma Gradio UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr
from frontend.gradio.obgyn.obgyn_ui import build_interface, GRADIO_PORT, CSS

if __name__ == "__main__":
    print(f"Starting GynecologistGemma UI on http://0.0.0.0:{GRADIO_PORT}")
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="violet"),
        css=CSS,
    )
