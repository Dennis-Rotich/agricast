"""USSD menu state machine (pure Python, no Streamlit). Simulated, not a live shortcode.

The plot comes from the phone's registration, so the farmer never types a plot code.
"""
from src.advisory.rule_engine import STAGES

DIAL_CODE = "*384*247#"
CROPS = list(STAGES)
MENU = "AgriCast\n1. Check plot risk\n0. Exit"


def _list_screen(title, items):
    return title + "\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))


def crop_menu():
    return _list_screen("Select crop:", CROPS)


def stage_menu(crop):
    return _list_screen("Select growth stage:", STAGES[crop])


def new_session(plot):
    return {"state": "menu", "plot": plot, "crop": None}


def _pick(text, items):
    t = text.strip()
    return items[int(t) - 1] if t.isdigit() and 1 <= int(t) <= len(items) else None


def step(session, text, advise):
    """advise(plot, crop, stage) -> text. Returns (screen_text, done)."""
    t, s = text.strip(), session["state"]
    if s == "menu":
        if t == "1":
            session["state"] = "crop"
            return crop_menu(), False
        if t == "0":
            return "Thank you. Goodbye.", True
        return "Invalid choice.\n" + MENU, False
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
