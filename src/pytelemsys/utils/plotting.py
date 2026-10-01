import mplcursors


def cursor_hover(artist, values) -> None:
    """Show the value associated with the hovered point of a plot.

    :param artist: matplotlib artist to annotate (e.g. a scatter).
    :param values: values to show, one for each point of the artist.
    """

    # Use mplcursors to show annotations on hover
    cursor = mplcursors.cursor(artist, hover=True)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(f"x: {values[int(sel.index)]:.2f}")
    )
