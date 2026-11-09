"""USSD menu state machine (pure Python, no Streamlit). Simulated, not a live shortcode."""
from src.advisory.rule_engine import STAGES

DIAL_CODE = "*384*247#"
CROPS = list(STAGES)
MENU = "AgriCast\n1. Check plot risk\n0. Exit"
PLOT_PROMPT = "Enter plot code (e.g. KAB_001):"


def _list_screen(title, items):
    return title + "\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))


def crop_menu():
    return _list_screen("Select crop:", CROPS)


def stage_menu(crop):
    return _list_screen("Select growth stage:", STAGES[crop])


def new_session():
    return {"state": "menu", "plot": None, "crop": None}


def _pick(text, items):
    t = text.strip()
    return items[int(t) - 1] if t.isdigit() and 1 <= int(t) <= len(items) else None


def step(session, text, plots, advise):
    """plots: {UPPERCASE code: real grid_id}. advise(plot, crop, stage) -> text.
    Returns (screen_text, done)."""
    t, s = text.strip(), session["state"]
    if s == "menu":
        if t == "1":
            session["state"] = "plot"
            return PLOT_PROMPT, False
        if t == "0":
            return "Thank you. Goodbye.", True
        return "Invalid choice.\n" + MENU, False
    if s == "plot":
        if t.upper() not in plots:
            return "Plot not found.\n" + PLOT_PROMPT, False
        session["plot"], session["state"] = plots[t.upper()], "crop"
        return crop_menu(), False
    if s == "crop":
        crop = _pick(t, CROPS)
        if crop is None:
            return "Invalid choice.\n" + crop_menu(), False
        session["crop"], session["state"] = crop, "stage"
        return stage_menu(crop), False
    if s == "stage":
        stage = _pick(t, STAGES[session["crop"]])
        if stage is None:
            return "Invalid choice.\n" + stage_menu(session["crop"]), False
        session["state"] = "result"
        return advise(session["plot"], session["crop"], stage) + "\n0. Menu", False
    if t == "0":
        session["state"] = "menu"
        return MENU, False
    return "Invalid choice.\n0. Menu", False
