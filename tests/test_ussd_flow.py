from src.advisory.rule_engine import STAGES, get_advisory
from src.advisory.ussd_flow import MENU, new_session, step

PLOTS = {"KAB_001": "KAB_001", "KAB_002": "KAB_002"}


def advise(plot, crop, stage):
    return get_advisory(crop, stage, 62, 14, "spei_30", plot)["sms"]


def run(*inputs):
    s, out = new_session(), []
    for i in inputs:
        out.append(step(s, i, PLOTS, advise))
    return s, out


def test_happy_path():
    s, out = run("1", "kab_001", "1", "1")
    screen, done = out[-1]
    assert "KAB_001" in screen and "MODERATE" in screen and not done
    assert s["state"] == "result"


def test_back_to_menu_and_exit():
    _, out = run("1", "KAB_001", "1", "1", "0", "0")
    assert out[-2][0] == MENU
    assert out[-1] == ("Thank you. Goodbye.", True)


def test_bad_inputs_do_not_advance():
    s, out = run("9", "1", "NOPE", "KAB_001", "7", "2", "5")
    assert out[0][0].startswith("Invalid choice")
    assert out[2][0].startswith("Plot not found")
    assert out[4][0].startswith("Invalid choice")
    assert out[6][0].startswith("Invalid choice") and s["state"] == "stage"


def test_every_screen_fits_ussd_limit():
    for crop, stages in STAGES.items():
        for i, _ in enumerate(stages, 1):
            ci = list(STAGES).index(crop) + 1
            _, out = run("1", "KAB_001", str(ci), str(i))
            for screen, _ in out:
                assert len(screen) <= 182, (len(screen), screen)
