"""make_results.py must never delete a section it cannot regenerate."""
import make_results


def test_merge_preserves_unknown_sections():
    old = "# T\n\nintro\n\n## Alpha\nold alpha\n\n## Handwritten\nprecious analysis\n"
    generated = [("Alpha", ["## Alpha", "new alpha", ""])]
    out = make_results.merge(old, ["# T", "", "intro", ""], generated)
    assert "## Handwritten" in out
    assert "precious analysis" in out          # the whole point of this plan
    assert "new alpha" in out                  # regenerated section was replaced
    assert "old alpha" not in out


def test_merge_keeps_old_order_and_appends_new():
    old = "# T\n\n## B\nb\n\n## A\na\n"
    generated = [("A", ["## A", "a2", ""]), ("C", ["## C", "c", ""])]
    out = make_results.merge(old, ["# T", ""], generated)
    assert out.index("## B") < out.index("## A") < out.index("## C")


def test_merge_on_empty_file_is_just_the_generated_sections():
    out = make_results.merge("", ["# T", ""], [("A", ["## A", "a", ""])])
    assert out.startswith("# T")
    assert "## A" in out
