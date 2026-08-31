"""Two different runs must never share each other's generated files.

Per-scene manifests live at fixed paths (data/images/manifest.json and so on)
rather than being namespaced per run, so resume has to prove a cached manifest
belongs to the poem currently being rendered."""
import json

from src import orchestration as orch


def _manifest(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"scene_index": i} for i in range(1, n + 1)]), encoding="utf-8")
    return path


def test_two_poems_of_the_same_length_do_not_share_images(tmp_path):
    """THE bug: scene count alone matched, so an 8-line run following another
    8-line run reused the first poem's images - the new words sung over the old
    pictures, silently."""
    poem_a = tmp_path / "a.json"
    poem_a.write_text('{"scenes": [1, 2, 3, 4, 5, 6, 7, 8]}', encoding="utf-8")
    poem_b = tmp_path / "b.json"
    poem_b.write_text('{"scenes": [8, 7, 6, 5, 4, 3, 2, 1]}', encoding="utf-8")

    manifest = _manifest(tmp_path / "images" / "manifest.json", 8)
    orch.stamp_manifest(manifest, orch.script_fingerprint(poem_a))

    assert orch.manifest_is_current(manifest, 8, orch.script_fingerprint(poem_a)), \
        "the poem that produced it must still be reusable"
    assert not orch.manifest_is_current(manifest, 8, orch.script_fingerprint(poem_b)), \
        "a different poem of the same length must NOT reuse these images"


def test_editing_a_poem_in_place_invalidates_its_images(tmp_path):
    """Same filename, changed wording - the images are of the old words."""
    script = tmp_path / "s.json"
    script.write_text('{"scenes": ["a lamb"]}', encoding="utf-8")
    manifest = _manifest(tmp_path / "images" / "manifest.json", 1)
    orch.stamp_manifest(manifest, orch.script_fingerprint(script))

    script.write_text('{"scenes": ["a pig"]}', encoding="utf-8")
    assert not orch.manifest_is_current(manifest, 1, orch.script_fingerprint(script))


def test_an_unstamped_manifest_is_stale(tmp_path):
    """Manifests written before this check existed cannot be proven to match,
    and assuming they do is the exact mistake being fixed."""
    script = tmp_path / "s.json"
    script.write_text('{"scenes": [1, 2]}', encoding="utf-8")
    manifest = _manifest(tmp_path / "images" / "manifest.json", 2)

    assert not orch.manifest_is_current(manifest, 2, orch.script_fingerprint(script))


def test_a_differing_scene_count_is_still_stale(tmp_path):
    """The original check must survive: a 16-scene manifest cannot serve a
    4-scene run even if something else matches."""
    script = tmp_path / "s.json"
    script.write_text('{"scenes": [1, 2, 3, 4]}', encoding="utf-8")
    manifest = _manifest(tmp_path / "images" / "manifest.json", 16)
    orch.stamp_manifest(manifest, orch.script_fingerprint(script))

    assert not orch.manifest_is_current(manifest, 4, orch.script_fingerprint(script))


def test_the_script_itself_is_keyed_to_the_input_poem(tmp_path):
    """Running a different poem under the same --name must not reuse the first
    poem's script - every later stage builds on it, so the whole video would be
    of the wrong poem."""
    poem_a = tmp_path / "a.txt"
    poem_a.write_text("A lamb walks down the lane\n", encoding="utf-8")
    poem_b = tmp_path / "b.txt"
    poem_b.write_text("A pig sits in the rain\n", encoding="utf-8")

    script = tmp_path / "scripts" / "shared_name.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('{"scenes": [1]}', encoding="utf-8")
    orch.stamp_manifest(script, orch.script_fingerprint(poem_a))

    assert orch.manifest_is_current(script, None, orch.script_fingerprint(poem_a))
    assert not orch.manifest_is_current(script, None, orch.script_fingerprint(poem_b))


def test_a_corrupt_manifest_is_stale(tmp_path):
    script = tmp_path / "s.json"
    script.write_text('{"scenes": [1]}', encoding="utf-8")
    manifest = tmp_path / "images" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{ not json", encoding="utf-8")

    assert not orch.manifest_is_current(manifest, 1, orch.script_fingerprint(script))
