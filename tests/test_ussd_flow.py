from src.advisory.rule_engine import STAGES, get_advisory
from src.advisory.ussd_flow import MENU, new_session, step


def advise(plot, crop, stage):
    return get_advisory(crop, stage, 62, 14, "spei_30", plot)["sms"]


def run(*inputs, plot="KAB_001"):
    s, out = new_session(plot), []
    for i in inputs:
        out.append(step(s, i, advise))
    return s, out


def test_happy_path_needs_no_plot_code():
    s, out = run("1", "1", "1")
    screen, done = out[-1]
    assert "KAB_001" in screen and "MODERATE" in screen and not done
    assert s["state"] == "result"


def test_plot_comes_from_the_session():
    _, out = run("1", "1", "1", plot="KAB_042")
    assert "KAB_042" in out[-1][0]


def test_back_to_menu_and_exit():
    _, out = run("1", "1", "1", "0", "0")
    assert out[-2][0] == MENU
    assert out[-1] == ("Thank you. Goodbye.", True)


def test_bad_inputs_do_not_advance():
    s, out = run("9", "1", "7", "2", "5")
    assert out[0][0].startswith("Invalid choice")
    assert out[2][0].startswith("Invalid choice")
    assert out[4][0].startswith("Invalid choice") and s["state"] == "stage"


def test_every_screen_fits_ussd_limit():
    for ci, (crop, stages) in enumerate(STAGES.items(), 1):
        for i, _ in enumerate(stages, 1):
            _, out = run("1", str(ci), str(i))
            for screen, _ in out:
                assert len(screen) <= 182, (len(screen), screen)
