import streamlit as st

from src.advisory.rule_engine import get_advisory, risk_driver
from src.advisory.ussd_flow import DIAL_CODE, MENU, new_session, step


def render(view, names, crop, stage, horizon):
    """view: forecasts already filtered to one date and horizon. names: Series grid_id -> place."""
    lookup = view.drop_duplicates("grid_id").set_index("grid_id")

    def advise(plot, c, s):
        r = lookup.loc[plot]
        return get_advisory(c, s, r.vulnerability_score_0_100, horizon,
                            riskdriver(r), plot)["sms"]

    left, right = st.columns(2)

    with left:
        st.subheader("SMS alert")
        plot = st.selectbox("Plot", list(lookup.index),
                            format_func=lambda g: f"{names.get(g, g)} ({g})")
        sms = advise(plot, crop, stage)
        st.code(sms, language=None)
        st.caption(f"{len(sms)} of 160 characters. Works on any phone, no data needed.")
        st.caption("Draft advice, pending agronomist review.")

    with right:
        st.subheader("USSD menu")
        ss = st.session_state
        ss.setdefault("ussd", None)
        ss.setdefault("ussd_screen", "")
        ss.setdefault("ussd_done", False)
        plots = {g.upper(): g for g in lookup.index}

        # No st.rerun() anywhere: state is updated first, then the screen is drawn.
        if st.button(f"Dial {DIAL_CODE}"):
            ss.ussd, ss.ussd_screen, ss.ussd_done = new_session(), MENU, False

        screen_box = st.container()   # drawn above the form, filled after input is handled
        form_box = st.empty()

        if ss.ussd is not None and not ss.ussd_done:
            with form_box.form("ussd_form", clear_on_submit=True):
                reply = st.text_input("Your reply")
                sent = st.form_submit_button("Send")
            if sent:
                ss.ussd_screen, ss.ussd_done = step(ss.ussd, reply, plots, advise)
                if ss.ussd_done:
                    form_box.empty()

        with screen_box:
            if ss.ussd is None:
                st.write("Press dial to start a session.")
            else:
                st.code(ss.ussd_screen, language=None)
                if ss.ussd_done:
                    st.caption("Session ended. Press dial to start again.")

        st.caption("Simulated code, not a live shortcode. Plot codes look like "
                   f"{next(iter(plots))}.")
