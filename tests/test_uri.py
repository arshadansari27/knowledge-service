from knowledge_service.ontology.uri import is_uri, slugify, to_entity_uri, to_predicate_uri, KS, KS_DATA


class TestIsUri:
    def test_http(self):
        assert is_uri("http://example.com") is True

    def test_https(self):
        assert is_uri("https://example.com") is True

    def test_urn(self):
        assert is_uri("urn:isbn:123") is True

    def test_bare_label(self):
        assert is_uri("cold_exposure") is False

    def test_empty(self):
        assert is_uri("") is False


class TestSlugify:
    def test_spaces(self):
        assert slugify("cold exposure") == "cold_exposure"

    def test_mixed_case(self):
        assert slugify("PostgreSQL") == "postgresql"

    def test_special_chars(self):
        assert slugify("vitamin D3") == "vitamin_d3"

    def test_multiple_underscores(self):
        assert slugify("a--b  c") == "a_b_c"

    def test_leading_trailing(self):
        assert slugify("__test__") == "test"

    def test_unicode(self):
        assert slugify("café") == "caf"


class TestToEntityUri:
    def test_bare_label(self):
        assert to_entity_uri("cold_exposure") == f"{KS_DATA}cold_exposure"

    def test_already_uri(self):
        assert to_entity_uri("http://example.com/e") == "http://example.com/e"

    def test_spaces_slugified(self):
        assert to_entity_uri("cold exposure") == f"{KS_DATA}cold_exposure"


class TestToPredicateUri:
    def test_bare_label(self):
        assert to_predicate_uri("causes") == f"{KS}causes"

    def test_already_uri(self):
        assert to_predicate_uri("http://example.com/p") == "http://example.com/p"

    def test_mixed_case_slugified(self):
        assert to_predicate_uri("leadTo") == f"{KS}leadto"
