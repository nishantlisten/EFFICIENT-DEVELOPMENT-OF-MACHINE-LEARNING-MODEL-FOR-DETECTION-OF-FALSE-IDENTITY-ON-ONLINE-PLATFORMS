"""
=============================================================
  Instagram Profile Scraper
=============================================================
Fetches public Instagram profile data using instaloader.
Extracts features needed for the fake profile ML model.
"""

import logging
import random
import re
import time

import instaloader
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
_REQUEST_TIMEOUT = 20  # seconds
_MAX_RETRIES = 3
_BACKOFF_BASE = 2      # seconds; delays are 2, 4, 8 …
_JITTER_MAX = 1.0      # seconds of random jitter added to each delay

# Realistic browser headers that Instagram expects from a real client.
# The x-ig-app-id is Instagram's public web app identifier.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.instagram.com/",
    "X-Requested-With": "XMLHttpRequest",
    "x-ig-app-id": "936619743392459",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def _make_session() -> requests.Session:
    """
    Build a requests.Session with connection pooling and a conservative
    urllib3 retry policy (network-level errors only — HTTP status retries
    are handled manually so we can apply exponential backoff with jitter).
    """
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)

    # Retry on connection-level failures only; HTTP 4xx/5xx are handled below.
    retry_policy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[],          # no automatic HTTP-status retries
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=4,
        pool_maxsize=8,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_with_backoff(session: requests.Session, url: str) -> requests.Response:
    """
    GET *url* using *session*, retrying up to _MAX_RETRIES times on
    HTTP 403 (IP block) or 429 (rate limit) responses.

    Delay schedule (seconds, before jitter):  2 → 4 → 8
    Jitter: uniform random in [0, _JITTER_MAX] added to each delay.

    Raises:
        requests.RequestException – on unrecoverable network errors.
    Returns:
        The last requests.Response received (caller checks status_code).
    """
    last_response = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info("[scraper] GET %s (attempt %d/%d)", url, attempt, _MAX_RETRIES)
            response = session.get(url, timeout=_REQUEST_TIMEOUT)
            last_response = response

            if response.status_code not in (403, 429):
                # Success or a definitive error (404, 500, …) — stop retrying.
                return response

            status = response.status_code
            reason = "IP blocked by Instagram" if status == 403 else "rate limited by Instagram"
            logger.warning(
                "[scraper] HTTP %d (%s) on attempt %d/%d for %s",
                status, reason, attempt, _MAX_RETRIES, url,
            )

            if attempt < _MAX_RETRIES:
                delay = (_BACKOFF_BASE ** attempt) + random.uniform(0, _JITTER_MAX)
                logger.info("[scraper] Backing off %.2fs before retry …", delay)
                time.sleep(delay)

        except requests.RequestException as exc:
            logger.error("[scraper] Network error on attempt %d: %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise
            delay = (_BACKOFF_BASE ** attempt) + random.uniform(0, _JITTER_MAX)
            time.sleep(delay)

    return last_response  # type: ignore[return-value]  # always set after ≥1 iteration


def extract_username_from_url(url: str) -> str:
    """
    Extract Instagram username from various URL formats.

    Supports:
      - https://www.instagram.com/username/
      - https://instagram.com/username
      - http://instagram.com/username/?hl=en
      - instagram.com/username
      - Just a username string
    """
    url = url.strip().rstrip("/")

    # Match Instagram URL patterns
    patterns = [
        r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9._]+)",
        r"^([A-Za-z0-9._]+)$",  # Plain username
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            username = match.group(1)
            # Filter out known Instagram paths that aren't usernames
            non_user_paths = {
                "p", "explore", "reels", "stories", "tv",
                "accounts", "directory", "about", "developer",
            }
            if username.lower() not in non_user_paths:
                return username

    return ""


def scrape_instagram_profile(url: str) -> dict:
    """
    Scrape public Instagram profile data.

    Strategy:
      1. Try Instagram's web_profile_info JSON API with full browser headers
         and exponential-backoff retries on 403/429 responses.
      2. Fall back to instaloader if the API is unavailable.

    Args:
        url: Instagram profile URL or username

    Returns:
        dict with profile data on success, or ``{"success": False, "error": "…"}``
    """
    username = extract_username_from_url(url)

    if not username:
        return {
            "success": False,
            "error": "Could not extract a valid username from the URL. "
                     "Please enter a valid Instagram profile URL "
                     "(e.g. https://instagram.com/username).",
        }

    logger.info("[scraper] Starting scrape for username: %s", username)

    session = _make_session()

    try:
        # ── 1. Instagram web_profile_info API ─────────────────
        api_url = (
            "https://www.instagram.com/api/v1/users/web_profile_info/"
            f"?username={username}"
        )

        response = _fetch_with_backoff(session, api_url)
        logger.info("[scraper] Final API response status: %d", response.status_code)

        if response.status_code == 200:
            data = response.json()
            user_data = data.get("data", {}).get("user", {})

            if user_data:
                fullname = user_data.get("full_name") or ""
                bio = user_data.get("biography") or ""
                num_followers = user_data.get("edge_followed_by", {}).get("count", 0)
                num_following = user_data.get("edge_follow", {}).get("count", 0)
                num_posts = user_data.get("edge_owner_to_timeline_media", {}).get("count", 0)

                profile_pic_url = user_data.get("profile_pic_url", "")
                has_profile_pic = bool(
                    profile_pic_url
                    and "44884218_345707102882519_2446069589734326272_n" not in profile_pic_url
                )

                external_url = bool(user_data.get("external_url"))
                is_private = bool(user_data.get("is_private"))
                is_verified = bool(user_data.get("is_verified"))

                logger.info(
                    "[scraper] Successfully scraped @%s via API "
                    "(followers=%d, posts=%d)",
                    username, num_followers, num_posts,
                )
                return {
                    "success": True,
                    "username": username,
                    "fullname": fullname,
                    "bio": bio,
                    "num_followers": num_followers,
                    "num_following": num_following,
                    "num_posts": num_posts,
                    "profile_pic": has_profile_pic,
                    "external_url": external_url,
                    "private": is_private,
                    "desc_length": len(bio),
                    "profile_pic_url": profile_pic_url,
                    "is_verified": is_verified,
                }

        elif response.status_code == 404:
            logger.warning("[scraper] Profile @%s not found (404)", username)
            return {
                "success": False,
                "error": f"Profile '@{username}' does not exist on Instagram.",
            }

        elif response.status_code == 403:
            logger.error(
                "[scraper] All retries exhausted — Instagram blocked requests "
                "from this IP (403). Falling back to instaloader."
            )

        elif response.status_code == 429:
            logger.error(
                "[scraper] All retries exhausted — Instagram rate-limited this "
                "IP (429). Falling back to instaloader."
            )

        else:
            logger.warning(
                "[scraper] Unexpected API status %d for @%s. "
                "Falling back to instaloader.",
                response.status_code, username,
            )

        # ── 2. Fallback: instaloader ──────────────────────────
        logger.info("[scraper] Attempting instaloader fallback for @%s", username)
        loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )

        profile = instaloader.Profile.from_username(loader.context, username)

        fullname = profile.full_name or ""
        bio = profile.biography or ""
        num_followers = profile.followers
        num_following = profile.followees
        num_posts = profile.mediacount
        has_profile_pic = (
            not profile.is_profile_pic_default()
            if hasattr(profile, "is_profile_pic_default")
            else bool(profile.profile_pic_url)
        )
        external_url = bool(profile.external_url)
        is_private = profile.is_private

        logger.info(
            "[scraper] Successfully scraped @%s via instaloader "
            "(followers=%d, posts=%d)",
            username, num_followers, num_posts,
        )
        return {
            "success": True,
            "username": username,
            "fullname": fullname,
            "bio": bio,
            "num_followers": num_followers,
            "num_following": num_following,
            "num_posts": num_posts,
            "profile_pic": has_profile_pic,
            "external_url": external_url,
            "private": is_private,
            "desc_length": len(bio),
            "profile_pic_url": profile.profile_pic_url,
            "is_verified": profile.is_verified,
        }

    except instaloader.exceptions.ProfileNotExistsException:
        logger.warning("[scraper] Profile @%s does not exist (instaloader)", username)
        return {
            "success": False,
            "error": f"Profile '@{username}' does not exist on Instagram.",
        }
    except instaloader.exceptions.ConnectionException as exc:
        error_msg = str(exc).lower()
        logger.error("[scraper] instaloader ConnectionException: %s", exc)
        if "login" in error_msg or "401" in error_msg:
            return {
                "success": False,
                "error": (
                    "Instagram requires login to view this profile. "
                    "The profile may be private or Instagram is rate-limiting "
                    "requests from this server. Please try again later or enter "
                    "profile details manually."
                ),
            }
        if "403" in error_msg:
            return {
                "success": False,
                "error": (
                    f"Instagram is blocking requests to '@{username}' from this "
                    "server's IP address (403 Forbidden). This is common on cloud "
                    "deployments. Please use the Manual Entry tab instead."
                ),
            }
        if "429" in error_msg:
            return {
                "success": False,
                "error": (
                    "Instagram is rate-limiting this server (429 Too Many Requests). "
                    "Please wait a few minutes and try again, or use the Manual Entry tab."
                ),
            }
        return {
            "success": False,
            "error": (
                f"Connection error while fetching '@{username}': {exc}. "
                "Instagram may be temporarily unavailable. Try again in a few minutes."
            ),
        }
    except requests.RequestException as exc:
        logger.error("[scraper] requests.RequestException: %s", exc)
        return {
            "success": False,
            "error": (
                f"Network error while contacting Instagram: {exc}. "
                "Please check your connection and try again."
            ),
        }
    except Exception as exc:
        logger.exception("[scraper] Unexpected error scraping @%s", username)
        return {
            "success": False,
            "error": (
                f"Failed to fetch profile '@{username}': {exc}. "
                "You can enter profile details manually instead."
            ),
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test URL parsing
    test_urls = [
        "https://www.instagram.com/meta/",
        "https://instagram.com/python",
        "instagram.com/elonmusk",
        "cristiano",
        "https://www.instagram.com/p/ABC123/",  # Should fail — it's a post
    ]

    print("=" * 50)
    print("  URL PARSING TESTS")
    print("=" * 50)

    for url in test_urls:
        username = extract_username_from_url(url)
        print(f"  {url:45s} → {username or '(invalid)'}")

    print("\n" + "=" * 50)
    print("  PROFILE SCRAPE TEST")
    print("=" * 50)

    result = scrape_instagram_profile("instagram")
    if result["success"]:
        for k, v in result.items():
            if k != "profile_pic_url":
                print(f"  {k:20s}: {v}")
    else:
        print(f"  Error: {result['error']}")
