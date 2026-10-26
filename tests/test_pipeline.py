import pandas as pd

from data_pipeline.generate_students import generate_students
from data_pipeline.import_form import students_from_form
from data_pipeline.load_kaggle import (derive_skills, is_bangalore, opportunities_from_dataframe,
                                       parse_duration_weeks, parse_stipend, synthetic_opportunities)


# ---- Kaggle loader -------------------------------------------------------

def test_bangalore_spellings():
    for text in ["Bangalore", "Bengaluru", "banglore, Karnataka", "Work From Home, Bengalore"]:
        assert is_bangalore(text)
    assert not is_bangalore("Mumbai")


def test_derive_skills_ml_title():
    required, preferred, domain = derive_skills("Machine Learning Intern (Python)")
    assert required == ["Python", "ML"]
    assert domain == "ML"
    assert preferred == ["Data analysis"]


def test_derive_skills_whole_words_only():
    required, _, domain = derive_skills("JavaScript Developer")
    assert "Java" not in required
    assert required == ["Web dev"] and domain == "Web"


def test_derive_skills_unknown_title():
    assert derive_skills("Intern")[0] == ["Communication"]


def test_parsers():
    assert parse_duration_weeks("3 Months") == 12
    assert parse_duration_weeks("6 Weeks") == 6
    assert parse_duration_weeks("13 Apr - 13 May' 24") == 8      # a date, not a duration
    assert parse_duration_weeks("Starts immediately") == 8
    assert parse_stipend("₹ 10,000 /month") == 10000
    assert parse_stipend("Unpaid") == 0


def test_dataframe_filter_dedupe_and_area():
    df = pd.DataFrame({
        "internship": ["ML Intern", "ML Intern", "Web Developer", "Data Analyst"],
        "company_name": ["A", "A", "B", "C"],
        "location": ["Bangalore", "Bangalore", "Bengaluru (Koramangala)", "Mumbai"],
        "duration": ["2 Months", "2 Months", "6 Weeks", "3 Months"],
        "stipend": ["5000 /month", "5000 /month", "Unpaid", "8000"],
    })
    opps = opportunities_from_dataframe(df)
    assert [o.title for o in opps] == ["ML Intern", "Web Developer"]   # Mumbai + duplicate dropped
    assert opps[1].area == "Koramangala"                               # area read from text
    assert all(o.source == "kaggle" for o in opps)


def test_domain_cap_keeps_scarce_tech_listings():
    titles = [f"Sales {i}" for i in range(10)] + ["Machine Learning"]
    df = pd.DataFrame({"Title": titles, "Company Name": [f"C{i}" for i in range(11)],
                       "Location": ["[['Bangalore']]"] * 11})
    opps = opportunities_from_dataframe(df, limit=5, max_per_domain=3)
    domains = [o.domain for o in opps]
    assert domains.count("Other") == 3 and "ML" in domains


def test_synthetic_opportunities_deterministic_and_has_koramangala_ml():
    a, b = synthetic_opportunities(), synthetic_opportunities()
    assert a == b
    assert any(o.domain == "ML" and o.area == "Koramangala" for o in a)


# ---- Synthetic students --------------------------------------------------

def test_students_deterministic():
    assert generate_students() == generate_students()
    assert len(generate_students()) == 150


def test_students_history_is_skewed():
    students = generate_students()
    past = sorted((s.past_opportunities for s in students), reverse=True)
    top_15 = past[: round(len(past) * 0.15)]
    assert sum(top_15) / sum(past) > 0.5


# ---- Google Form import --------------------------------------------------

def test_form_import_drops_non_consent_and_maps_levels():
    skills = ["Python", "Java", "ML", "Data analysis", "Web dev", "App dev",
              "Cloud/DevOps", "Design", "Communication", "Leadership"]
    base = {"Anonymous ID": "x1", "Branch": "CSE", "Year": "3",
            "Home area in Bangalore": "Koramangala 5th block",
            **{s: "None" for s in skills},
            "Interests": "ML, Web", "Goal": "Internship", "Hours available per week": "12",
            "Internships/projects received so far": "3+", "Consent": "I agree"}
    row_ok = {**base, "ML": "Advanced", "Python": "Beginner"}
    row_no = {**base, "Anonymous ID": "x2", "Consent": ""}
    students = students_from_form(pd.DataFrame([row_ok, row_no]))

    assert len(students) == 1
    s = students[0]
    assert s.id == "F-x1" and s.source == "form"
    assert s.skills["ML"] == 3 and s.skills["Python"] == 1 and s.skills["Java"] == 0
    assert s.home_area == "Koramangala"
    assert s.interests == ["ML", "Web"]
    assert s.past_opportunities == 3 and s.hours_per_week == 12
