import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import youtube_downloader
from youtube_downloader import (
    _candidate_match_filter,
    _candidate_is_eligible,
    _candidate_priority,
    _select_candidates,
    has_allowed_license,
)


class AllowedLicenseTests(unittest.TestCase):
    def test_main_reconfigures_streams_before_printing_unicode_json(self) -> None:
        class ReconfigurableStringIO(io.StringIO):
            def reconfigure(self, **settings):
                self.encoding_settings = settings

        stdout = ReconfigurableStringIO()
        stderr = ReconfigurableStringIO()
        with (
            patch.object(
                youtube_downloader.sys,
                "argv",
                ["youtube_downloader.py", "--keyword", "ceramica"],
            ),
            patch.object(youtube_downloader.sys, "stdout", stdout),
            patch.object(youtube_downloader.sys, "stderr", stderr),
            patch.object(
                youtube_downloader,
                "download_matches",
                return_value=[
                    {
                        "id": "video12345",
                        "title": "Una tazza 🎬",
                        "url": "https://youtube.com/watch?v=video12345",
                        "license": "None",
                        "file": "video12345.mp4",
                    }
                ],
            ),
        ):
            result_code = youtube_downloader.main()

        self.assertEqual(result_code, 0)
        self.assertEqual(stdout.encoding_settings, {
            "encoding": "utf-8",
            "errors": "backslashreplace",
        })
        self.assertEqual(stderr.encoding_settings, {
            "encoding": "utf-8",
            "errors": "backslashreplace",
        })
        self.assertEqual(json.loads(stdout.getvalue())["videos"][0]["title"], "Una tazza 🎬")

    def test_accepts_youtube_creative_commons_attribution_license(self) -> None:
        self.assertTrue(
            has_allowed_license("Creative Commons Attribution license (reuse allowed)")
        )

    def test_rejects_missing_or_unrecognized_license(self) -> None:
        self.assertFalse(has_allowed_license(None))
        self.assertFalse(has_allowed_license("Standard YouTube License"))
        self.assertFalse(has_allowed_license("Creative Commons Attribution"))

    def test_select_candidates_fills_limit_from_lower_priority_tiers(self) -> None:
        ideal = {
            "id": "ideal12345",
            "webpage_url": "https://youtube.com/watch?v=ideal12345",
            "duration": 450,
            "width": 1920,
            "height": 1080,
        }
        standard = [
            {
                "id": f"standard{index:02d}",
                "webpage_url": f"https://youtube.com/watch?v=standard{index:02d}",
                "duration": 720,
                "width": 1920,
                "height": 1080,
            }
            for index in range(1, 10)
        ]

        selected = _select_candidates([*standard, ideal], 10)

        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[0], ideal)

    def test_main_accepts_ten_video_candidates(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(
                youtube_downloader.sys,
                "argv",
                ["youtube_downloader.py", "--keyword", "ceramica", "--limit", "10"],
            ),
            patch.object(youtube_downloader.sys, "stdout", stdout),
            patch.object(youtube_downloader.sys, "stderr", stderr),
            patch.object(youtube_downloader, "download_matches", return_value=[]) as download,
        ):
            result_code = youtube_downloader.main()

        self.assertEqual(result_code, 0)
        download.assert_called_once()
        self.assertEqual(download.call_args.args[:2], ("ceramica", 10))
        self.assertTrue(callable(download.call_args.args[2]))

    def test_main_can_export_browser_cookies_without_a_search_keyword(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(
                youtube_downloader.sys,
                "argv",
                ["youtube_downloader.py", "--export-cookies-from-browser", "firefox"],
            ),
            patch.object(youtube_downloader.sys, "stdout", stdout),
            patch.object(youtube_downloader.sys, "stderr", stderr),
            patch.object(youtube_downloader, "export_browser_cookies", return_value=3) as export,
        ):
            result_code = youtube_downloader.main()

        self.assertEqual(result_code, 0)
        export.assert_called_once_with("firefox")
        self.assertEqual(json.loads(stdout.getvalue())["cookie_count"], 3)
        self.assertEqual(stderr.getvalue(), "")

    def test_one_download_failure_does_not_stop_later_candidates(self) -> None:
        attempted_downloads = []
        video_metadata = {
            video_id: {
                "id": video_id,
                "title": video_id,
                "webpage_url": f"https://youtube.com/watch?v={video_id}",
                "duration": 450,
                "width": 1920,
                "height": 1080,
            }
            for video_id in ("first12345", "second1234")
        }

        class FakeYouTubeDL:
            def __init__(self, _options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return not exc_info

            def extract_info(self, url, download=False):
                if url.startswith("ytsearch"):
                    return {
                        "entries": [
                            {"webpage_url": metadata["webpage_url"]}
                            for metadata in video_metadata.values()
                        ]
                    }

                video_id = url.rsplit("=", 1)[-1]
                metadata = dict(video_metadata[video_id])
                if not download:
                    return metadata
                attempted_downloads.append(video_id)
                if video_id == "first12345":
                    raise youtube_downloader.yt_dlp.utils.DownloadError("non disponibile")
                metadata["filepath"] = f"{video_id}.mp4"
                return metadata

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(youtube_downloader, "DOWNLOAD_DIR", Path(temporary)),
                patch.object(youtube_downloader.yt_dlp, "YoutubeDL", FakeYouTubeDL),
            ):
                videos = youtube_downloader.download_matches("ceramica", 2)

        self.assertEqual(attempted_downloads, ["first12345", "second1234"])
        self.assertEqual([video["id"] for video in videos], ["second1234"])

    def test_candidate_filters_duration_shorts_and_portrait_orientation(self) -> None:
        base = {
            "webpage_url": "https://youtube.com/watch?v=video12345",
            "duration": 300,
            "width": 1280,
            "height": 720,
        }
        self.assertTrue(_candidate_is_eligible(base))
        self.assertTrue(
            _candidate_is_eligible(
                {**base, "duration": 60, "height": None, "width": None}
            )
        )
        self.assertTrue(
            _candidate_is_eligible(
                {**base, "duration": 900, "height": None, "width": None}
            )
        )
        self.assertFalse(_candidate_is_eligible({**base, "duration": 59}))
        self.assertFalse(_candidate_is_eligible({**base, "duration": 901}))
        self.assertFalse(
            _candidate_is_eligible(
                {**base, "webpage_url": "https://youtube.com/shorts/video12345"}
            )
        )
        self.assertFalse(_candidate_is_eligible({**base, "width": 720, "height": 1280}))

    def test_candidate_priority_prefers_horizontal_ideal_duration(self) -> None:
        ideal_landscape = {
            "duration": 450,
            "width": 1920,
            "height": 1080,
        }
        landscape_outside_ideal = {
            "duration": 720,
            "width": 1920,
            "height": 1080,
        }
        ideal_unknown_orientation = {"duration": 450}

        self.assertGreater(
            _candidate_priority(ideal_landscape),
            _candidate_priority(landscape_outside_ideal),
        )
        self.assertGreater(
            _candidate_priority(ideal_landscape),
            _candidate_priority(ideal_unknown_orientation),
        )

    def test_candidate_filter_bypasses_license_and_duration_checks(self) -> None:
        self.assertIsNone(
            _candidate_match_filter(
                {
                    "webpage_url": "https://youtube.com/watch?v=video12345",
                    "duration": 1200,
                    "license": None,
                },
                incomplete=False,
            )
        )

    def test_selection_falls_back_to_the_shortest_horizontal_video_over_15_minutes(self) -> None:
        candidates = [
            {
                "webpage_url": "https://youtube.com/watch?v=huge12345",
                "duration": 7200,
                "width": 1920,
                "height": 1080,
            },
            {
                "webpage_url": "https://youtube.com/watch?v=shortest1",
                "duration": 1200,
                "width": 1920,
                "height": 1080,
            },
            {
                "webpage_url": "https://youtube.com/watch?v=portrait1",
                "duration": 420,
                "width": 720,
                "height": 1280,
            },
        ]

        selected = _select_candidates(candidates, 1)

        self.assertEqual(selected, [candidates[1]])

    def test_selection_uses_one_to_fifteen_minute_videos_before_longer_videos(self) -> None:
        within_limit = {
            "webpage_url": "https://youtube.com/watch?v=within12",
            "duration": 720,
            "width": 1920,
            "height": 1080,
        }
        too_long = {
            "webpage_url": "https://youtube.com/watch?v=over900",
            "duration": 1200,
            "width": 1920,
            "height": 1080,
        }

        selected = _select_candidates([too_long, within_limit], 1)

        self.assertEqual(selected, [within_limit])

    def test_selection_uses_first_non_shorts_result_as_last_resort(self) -> None:
        candidates = [
            {
                "webpage_url": "https://youtube.com/shorts/shorts123",
                "duration": 45,
            },
            {
                "webpage_url": "https://youtube.com/watch?v=portrait1",
                "duration": 420,
                "width": 720,
                "height": 1280,
            },
            {
                "webpage_url": "https://youtube.com/watch?v=portrait2",
                "duration": 600,
                "width": 720,
                "height": 1280,
            },
        ]

        selected = _select_candidates(candidates, 1)

        self.assertEqual(selected, [candidates[1]])

    def test_search_verifies_full_metadata_and_reports_download_progress(self) -> None:
        class FakeYouTubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return not exc_info

            def extract_info(self, url, download=False):
                if url.startswith("ytsearch"):
                    return {
                        "entries": [
                            {"webpage_url": "https://youtube.com/watch?v=bad12345"},
                            {"webpage_url": "https://youtube.com/watch?v=abc12345"},
                        ]
                    }
                if not download:
                    return {
                        "id": url.rsplit("=", 1)[-1],
                        "title": "Ceramica al tornio",
                        "webpage_url": url,
                        "license": (
                            "Standard YouTube License"
                            if "bad12345" in url
                            else "Creative Commons Attribution license (reuse allowed)"
                        ),
                        "duration": 420,
                        "width": 1280,
                        "height": 720,
                    }

                for hook in self.options.get("progress_hooks", []):
                    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                    hook({"status": "finished"})
                return {
                    "id": "abc12345",
                    "title": "Ceramica al tornio",
                    "webpage_url": url,
                    "license": "Creative Commons Attribution license (reuse allowed)",
                    "filepath": "abc12345.mp4",
                }

        with tempfile.TemporaryDirectory() as temporary:
            events = []
            with (
                patch.object(youtube_downloader, "DOWNLOAD_DIR", Path(temporary)),
                patch.object(youtube_downloader.yt_dlp, "YoutubeDL", FakeYouTubeDL),
            ):
                videos = youtube_downloader.download_matches(
                    "ceramica artigianale",
                    1,
                    events.append,
                )

        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["id"], "abc12345")
        self.assertEqual(videos[0]["license"], "Creative Commons Attribution license (reuse allowed)")
        self.assertTrue(any(event["stage"] == "search" for event in events))
        self.assertTrue(any(event["stage"] == "download" for event in events))
        self.assertTrue(any(event["stage"] == "download" and event["stage_progress"] == 100 for event in events))

    def test_search_filters_before_download_and_sorts_ideal_landscape_first(self) -> None:
        metadata_by_id = {
            "long12345": {
                "duration": 720,
                "width": 1920,
                "height": 1080,
            },
            "short12345": {
                "duration": 45,
                "width": 1920,
                "height": 1080,
            },
            "vertical1": {
                "duration": 420,
                "width": 720,
                "height": 1280,
            },
            "toolong12": {
                "duration": 901,
                "width": 1920,
                "height": 1080,
            },
            "ideal12345": {
                "duration": 420,
                "width": 1280,
                "height": 720,
            },
        }
        downloaded_ids = []

        class FakeYouTubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return not exc_info

            def extract_info(self, url, download=False):
                if url.startswith("ytsearch"):
                    return {
                        "entries": [
                            {"webpage_url": f"https://youtube.com/watch?v={video_id}"}
                            for video_id in metadata_by_id
                        ]
                        + [
                            {
                                "webpage_url": "https://youtube.com/shorts/short12345"
                            }
                        ]
                    }

                video_id = url.rsplit("=", 1)[-1]
                metadata = metadata_by_id[video_id]
                result = {
                    **metadata,
                    "id": video_id,
                    "title": video_id,
                    "webpage_url": url,
                    "license": "Creative Commons Attribution license (reuse allowed)",
                }
                if download:
                    match_filter = self.options["match_filter"]
                    self.assert_candidate_allowed(match_filter, result)
                    downloaded_ids.append(video_id)
                    result["filepath"] = f"{video_id}.mp4"
                return result

            @staticmethod
            def assert_candidate_allowed(match_filter, result):
                if match_filter(result, incomplete=False):
                    raise AssertionError("A filtered candidate reached the download stage.")

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(youtube_downloader, "DOWNLOAD_DIR", Path(temporary)),
                patch.object(youtube_downloader.yt_dlp, "YoutubeDL", FakeYouTubeDL),
            ):
                videos = youtube_downloader.download_matches(
                    "pressa idraulica",
                    1,
                )

        self.assertEqual(downloaded_ids, ["ideal12345"])
        self.assertEqual([video["id"] for video in videos], ["ideal12345"])

    def test_cookie_file_is_passed_to_search_metadata_and_download_calls(self) -> None:
        captured_options = []
        metadata = {
            "id": "cookie12345",
            "title": "Video con sessione",
            "webpage_url": "https://youtube.com/watch?v=cookie12345",
            "duration": 420,
            "width": 1280,
            "height": 720,
            "filepath": "cookie12345.mp4",
        }

        class FakeYouTubeDL:
            def __init__(self, options):
                captured_options.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return not exc_info

            def extract_info(self, url, download=False):
                if url.startswith("ytsearch"):
                    return {"entries": [{"webpage_url": metadata["webpage_url"]}]}
                return dict(metadata)

        with tempfile.TemporaryDirectory() as temporary:
            cookie_file = Path(temporary) / "youtube_cookies.txt"
            cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            with (
                patch.dict(
                    youtube_downloader.os.environ,
                    {"YOUTUBE_COOKIES_FILE": str(cookie_file)},
                ),
                patch.object(youtube_downloader, "DOWNLOAD_DIR", Path(temporary) / "downloads"),
                patch.object(youtube_downloader.yt_dlp, "YoutubeDL", FakeYouTubeDL),
            ):
                videos = youtube_downloader.download_matches("ceramica", 1)

        self.assertEqual(len(videos), 1)
        self.assertEqual(len(captured_options), 3)
        self.assertTrue(
            all(options.get("cookiefile") == str(cookie_file) for options in captured_options)
        )

    def test_exports_browser_cookies_to_a_private_netscape_cache(self) -> None:
        class FakeCookieJar:
            def save(self, filename, *, ignore_discard, ignore_expires):
                self.assert_save_options(ignore_discard, ignore_expires)
                Path(filename).write_text(
                    "# Netscape HTTP Cookie File\n"
                    "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSID\ttest-value\n",
                    encoding="utf-8",
                )

            @staticmethod
            def assert_save_options(ignore_discard, ignore_expires):
                if not ignore_discard or not ignore_expires:
                    raise AssertionError("Session cookies must be included in the exported cache.")

        with tempfile.TemporaryDirectory() as temporary:
            cookie_file = Path(temporary) / "data" / "youtube_cookies.txt"
            with (
                patch.object(youtube_downloader, "youtube_cookies_file", return_value=cookie_file),
                patch.object(
                    youtube_downloader,
                    "extract_cookies_from_browser",
                    return_value=FakeCookieJar(),
                ) as extract,
            ):
                cookie_count = youtube_downloader.export_browser_cookies("chrome")

            self.assertEqual(cookie_count, 1)
            self.assertEqual(extract.call_args.args, ("chrome",))
            self.assertIn("test-value", cookie_file.read_text(encoding="utf-8"))
            if youtube_downloader.os.name != "nt":
                self.assertEqual(cookie_file.stat().st_mode & 0o777, 0o600)

    def test_browser_cookie_database_lock_has_a_clear_fallback_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    youtube_downloader,
                    "youtube_cookies_file",
                    return_value=Path(temporary) / "youtube_cookies.txt",
                ),
                patch.object(
                    youtube_downloader,
                    "extract_cookies_from_browser",
                    side_effect=OSError("database is locked"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Chiudilo completamente"):
                    youtube_downloader.export_browser_cookies("chrome")

    def test_age_restriction_errors_prompt_for_cached_youtube_login(self) -> None:
        message = youtube_downloader.friendly_download_error(
            "Sign in to confirm your age. Use --cookies-from-browser or --cookies."
        )

        self.assertIn("Login YouTube (Cache)", message)
        self.assertTrue(youtube_downloader.youtube_auth_required(message))


if __name__ == "__main__":
    unittest.main()
