import mplcursors

def cursor_hover(colorbar, axis) -> None:
    """
    Add hover annotations to a colorbar."
    """
    
    # Use mplcursors to show annotations on hover
    cursor = mplcursors.cursor(colorbar, hover=True)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(
            f"x: {axis[sel.index]:.2f}"
        )
    )

