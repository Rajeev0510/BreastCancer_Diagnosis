import sys
print("=== BC APP: import started ===", flush=True)

import streamlit as st
print("=== BC APP: streamlit imported ===", flush=True)
# merged_app.py
import os
import io
import runpy
import streamlit as st

st.set_page_config(page_title="Combined App", layout="wide")

st.title("Combined Streamlit App")
st.caption("Two apps, one window — switch tabs to view each.")

# Locate the two scripts (update the paths if you move this file)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BC_IHC_PATH = os.path.join(THIS_DIR, "bc_ihc_onepage.py")
MAIN_APP_PATH = os.path.join(THIS_DIR, "app.py")

# Some Streamlit apps call st.set_page_config() internally; calling it twice causes an error.
# We patch it to a no-op *inside* the executed scripts to avoid exceptions.
class _NoOp:
    def __call__(self, *args, **kwargs):
        pass

def _safe_run_script(path: str, tab_label: str):
    """Run a Streamlit script in an isolated namespace, with safety guards."""
    st.subheader(tab_label)

    # Prepare an isolated global namespace.
    # We pass a patched 'st' that disables set_page_config inside the child script.
    import types
    ns = {
        "__name__": "__streamlit_tab__",
        "__file__": path,
        "st": st,
    }

    # Patch set_page_config to avoid double-call issues within the executed script.
    original_set_page_config = getattr(st, "set_page_config", None)
    try:
        st.set_page_config = _NoOp()  # type: ignore[attr-defined]
        # Run the target script
        try:
            runpy.run_path(path, init_globals=ns)
        except FileNotFoundError:
            st.error(f"File not found: {path}")
        except Exception as e:
            import traceback
            st.error(f"An error occurred while running `{os.path.basename(path)}`:\n\n{e}")
            st.exception(e)
    finally:
        # Restore original set_page_config
        if original_set_page_config is not None:
            st.set_page_config = original_set_page_config  # type: ignore[attr-defined]

tab1, tab2 = st.tabs(["BC IHC", "Main App"])

with tab1:
    _safe_run_script(BC_IHC_PATH, "BC IHC")

with tab2:
    _safe_run_script(MAIN_APP_PATH, "Main App")
