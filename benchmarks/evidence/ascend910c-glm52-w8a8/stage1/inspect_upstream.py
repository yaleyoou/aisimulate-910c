import concurrent.futures
import datetime
import json
import pathlib
import urllib.request


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "AISimulate-stage1-audit"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def main():
    release_url = "https://api.github.com/repos/sgl-project/sglang/releases/latest"
    registry_url = "https://quay.io/api/v1/repository/ascend/sglang/tag/?limit=100&onlyActiveTags=true"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        release, registry = executor.map(fetch, (release_url, registry_url))
    selected_tags = [
        {key: tag.get(key) for key in ("name", "manifest_digest", "last_modified", "size")}
        for tag in registry["tags"]
        if "a3" in tag["name"] and (
            tag["name"].startswith(release["tag_name"] + "-")
            or tag["name"].startswith("main-")
        )
    ]
    data = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_source": release_url,
        "release": {key: release.get(key) for key in ("tag_name", "published_at", "html_url", "prerelease", "draft", "target_commitish")},
        "registry_source": registry_url,
        "registry_has_additional_pages": registry.get("has_additional"),
        "selected_tags": selected_tags,
        "selection_rule": "Latest upstream stable release with an available matching Ascend A3 image; moving main tags excluded from the baseline.",
        "limits": "Registry tag metadata alone does not establish image package versions or driver compatibility.",
    }
    output = pathlib.Path(__file__).with_name("upstream-version-check.json")
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
