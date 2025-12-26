# streamlit_app.py
# Combined Streamlit launcher for two sub-app scripts (BC_IHC + Main App)

import os
import io
import runpy
import contextlib

print("=== BC APP: import started ===", flush=True)
import streamlit as st
print("=== BC APP: streamlit imported ===", flush=True)


def _safe_run_script(path: str, label: str) -> None:
    """
    Run a Streamlit script in-process, safely.

    - Friendly error if the file is missing.
    - Captures stdout/stderr and displays it (helps debugging on Streamlit Cloud).
    - Temporarily replaces st.set_page_config in the child script to prevent nested config errors.
    """
    if not os.path.exists(path):
        st.error(
            f"❌ `{label}` file not found: `{path}`\n\n"
            "Make sure this file exists in the GitHub repo "
            "(Streamlit Cloud cannot see your local Mac files)."
        )
        return

    st.subheader(label)

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    original_set_page_config = getattr(st, "set_page_config", None)

    def _no_op_set_page_config(*args, **kwargs):
        return None

    try:
        # prevent child scripts from calling set_page_config and crashing the combined app
        st.set_page_config = _no_op_set_page_config  # type: ignore[assignment]

        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            runpy.run_path(path, run_name="__main__")

    except Exception as e:
        st.error(f"⚠️ Error while running `{os.path.basename(path)}`: {e}")
        st.exception(e)

    finally:
        if original_set_page_config is not None:
            st.set_page_config = original_set_page_config  # type: ignore[assignment]

    stdout = out_buf.getvalue().strip()
    stderr = err_buf.getvalue().strip()
    if stdout or stderr:
        with st.expander(f"Debug output for {label}", expanded=False):
            if stdout:
                st.markdown("**stdout**")
                st.code(stdout)
            if stderr:
                st.markdown("**stderr**")
                st.code(stderr)


def main() -> None:
    st.set_page_config(page_title="BC Diagnosis Portal", layout="wide")

    st.title("BC Diagnosis Portal")
    st.caption("Two apps, one window — switch tabs to view each. (BC_IHC + BC_ultrasound/Main)")

    this_dir = os.path.dirname(os.path.abspath(__file__))

    # ✅ Update these filenames if your repo uses different names
    bc_ihc_path = os.path.join(this_dir, "bc_ihc_onepage.py")
    main_app_path = os.path.join(this_dir, "app.py")

    # Optional overrides via Streamlit Cloud Environment Variables
    bc_ihc_path = os.getenv("BC_IHC_PATH", bc_ihc_path)
    main_app_path = os.getenv("MAIN_APP_PATH", main_app_path)

    # Helpful on-page debug to confirm paths in Cloud
    with st.expander("Path debug (click to expand)"):
        st.write("Repo directory:", this_dir)
        st.write("BC IHC path:", bc_ihc_path)
        st.write("Main app path:", main_app_path)
        st.write("BC IHC exists:", os.path.exists(bc_ihc_path))
        st.write("Main app exists:", os.path.exists(main_app_path))

    tab1, tab2 = st.tabs(["BC IHC", "Main App"])

    with tab1:
        _safe_run_script(bc_ihc_path, "BC IHC")

    with tab2:
        _safe_run_script(main_app_path, "Main App")


if __name__ == "__main__":
    print("=== BC APP: entering main() ===", flush=True)
    main()
