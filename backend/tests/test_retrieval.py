from app.services.search_service import SearchService


def test_keyword_score():
    terms = SearchService._terms("What is the room rent limit?")
    assert "room" in terms
    assert "rent" in terms
    assert "limit" in terms


def test_keyword_score_matches_terms():
    score = SearchService._keyword_score(
        "The room rent limit is 5000 per day.",
        ["room", "rent", "limit"],
    )
    assert score == 1.0
